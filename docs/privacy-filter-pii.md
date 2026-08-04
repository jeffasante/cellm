# Running a PII Detector On-Device: openai/privacy-filter in cellm

**4 August 2026**

A tool that finds personal information in text should never send that text
anywhere. That is the whole argument for running this model locally: if you
upload the document to check whether it contains someone's phone number, you
have already leaked the phone number.

This is the story of taking `openai/privacy-filter` — a 2.6 GB model published
by OpenAI — and shrinking it to something that runs on a laptop or phone in
Rust, with no Python and no network.

## What the model does

You give it a sentence. It tells you which parts are personal information, and
what kind.

```
=== Contact Bob Smith at bob.smith@example.com or (555) 123-4567.
    private_person   [   7:  17]  " Bob Smith"
    private_email    [  20:  42]  " bob.smith@example.com"
    private_phone    [  45:  60]  " (555) 123-4567"
```

The numbers in brackets are exact character positions, so blanking out the
sensitive parts is a straightforward substring replacement.

It recognises eight categories:

| Category | Example |
| --- | --- |
| `private_person` | Bob Smith |
| `private_email` | bob@example.com |
| `private_phone` | (555) 123-4567 |
| `private_address` | 1600 Pennsylvania Ave |
| `private_date` | 14 February 1989 |
| `account_number` | 0040163411018 |
| `private_url` | https://acme.example/bk/9912 |
| `secret` | an API key or password |

## Why it starts out so big

The model is a **mixture of experts**. Rather than one large network that
processes every word, it holds 128 small specialist networks per layer and picks
the best four for each word.

```mermaid
graph LR
    Word["A word arrives"] --> Router["Router picks 4 of 128"]
    Router --> E1["Expert 12"]
    Router --> E2["Expert 47"]
    Router --> E3["Expert 88"]
    Router --> E4["Expert 103"]
    E1 --> Out["Combined result"]
    E2 --> Out
    E3 --> Out
    E4 --> Out
```

This is fast, because only 4 of the 128 do any work. But all 128 must still be
*stored*, which is why the file is large. Multiply 128 experts by 8 layers and
you get 1,024 expert networks sitting on disk — **84% of the entire file**.

## Making it smaller

Every number in the model is originally stored in 32 bits. That is far more
precision than the model needs. Quantization keeps fewer bits per number.

```mermaid
graph LR
    A["32 bits per number<br/>2.6 GB"] --> B["4 bits per number<br/>899 MB"]
    B --> C["3 bits per number<br/>667 MB"]
    C --> D["2 bits per number<br/>599 MB"]
```

The trick that makes this work is grouping. Numbers are handled 32 at a time.
For each group of 32 we record the smallest value and the size of one step, then
store each number as "how many steps above the smallest". With 4 bits you get 16
possible steps; with 2 bits, only 4.

Fewer steps means coarser approximation, which means mistakes.

## How much accuracy is lost

To measure this honestly, the original 2.6 GB model was run over 140 test
sentences and its answers were treated as the correct ones. Then each shrunken
version was run over the same sentences. **"Leak" is the percentage of personal
information the original found but the small version missed.**

| Build | Size | Recall | Precision | F1 | Leak |
| --- | --- | --- | --- | --- | --- |
| **4-bit, groups of 32** | **899 MB** | **0.978** | **0.975** | **0.977** | **2.2%** |
| 3-bit, groups of 128 | 667 MB | 0.954 | 0.923 | 0.938 | 4.6% |
| 2-bit, groups of 32 | 599 MB | 0.934 | 0.919 | 0.927 | 6.6% |

Halving the file roughly triples the misses. But the average hides the
interesting detail — some categories degrade far faster than others.

| Category | 4-bit | 3-bit | 2-bit |
| --- | --- | --- | --- |
| Names | 1.000 | 1.000 | 1.000 |
| Phone numbers | 1.000 | 1.000 | 1.000 |
| Email addresses | 1.000 | 1.000 | 0.954 |
| Web addresses | 1.000 | 0.923 | **0.769** |
| Street addresses | 0.952 | **0.833** | 0.881 |
| Account numbers | 0.943 | 0.914 | **0.829** |
| Dates | 0.933 | 0.911 | 0.867 |

Names and phone numbers are essentially indestructible. Structured identifiers
are fragile. At 2 bits, nearly a quarter of web addresses and a sixth of account
numbers go undetected.

### The failure that matters most

The 2-bit version does something worse than missing information: it **finds part
of it**. Asked about `bob.smith@example.com`, it may return only `.com`.

```mermaid
graph TD
    In["bob.smith@example.com"] --> Good["4-bit: full address found"]
    In --> Bad["2-bit: only '.com' found"]
    Good --> R1["Redacted safely"]
    Bad --> R2["bob.smith@example redacted<br/>— name still visible"]
```

A partial match looks like a success to whatever code called the model. Nothing
raises an error. The redaction runs, reports that it worked, and the person's
name stays in the document.

## Memory versus file size

A large file does not necessarily mean large memory use. The weights are
memory-mapped, meaning the operating system loads pages from disk only when they
are actually read — and since only 4 of 128 experts run per word, most of the
file is never touched.

| Build | Startup + one query | Peak memory |
| --- | --- | --- |
| 4-bit, 899 MB file | 0.92 s | **479 MB** |
| 3-bit, 667 MB file | 0.78 s | 440 MB |
| 2-bit, 599 MB file | 0.75 s | 432 MB |

Measured on an Apple M4, CPU only.

**The 899 MB model runs in 479 MB of memory.** Dropping to 2 bits saves 47 MB of
actual memory while tripling the miss rate. The smaller builds only make sense
if the constraint is app download size, not RAM.

## What it still gets wrong

These are limitations of the model itself and appear in every version:

- **Card security codes (CVV) are not detected at all.**
- **Bank SWIFT/BIC codes are not detected at all.**
- Bank sort codes such as `11-01-04` are labelled as dates.
- IP addresses are labelled as web addresses.
- In text dense with personal data, company names get flagged as addresses, so
  expect some over-redaction of organisations.

There is also a counter-intuitive behaviour worth knowing about: the model
correctly **ignores documentation placeholders**. The well-known example AWS key
`AKIAIOSFODNN7EXAMPLE` scores as "not a secret" with near-total confidence,
while a realistic-looking key is caught immediately. Testing with placeholder
credentials will make the model look broken when it is working exactly as
intended.

## Was a smaller model an option?

Two alternatives were tested.

`gravitee-io/bert-small-pii-detection` is only 110 MB and covers categories this
model lacks — social security numbers, passports, credit cards, IBANs. But it
breaks names into fragments, truncates email addresses (returning
`team@company.com` for `finance-team@company.com`), and missed `John Doe`
entirely in several sentences. Truncated personal information is still leaked
personal information.

Cutting the expert count was also investigated. Measuring which experts actually
activate across the test corpus showed the router spreads work evenly — keeping
only the top 16 would discard about half of all routing decisions in every
layer. There is no dead weight to remove.

Reaching 100 MB would require training a new, smaller model using this one as a
teacher. That remains the only viable path.

## Conclusion

**Use the 4-bit, group-32 build at 899 MB.** It stays within 2.2% of the
original 2.6 GB model, runs in under half a gigabyte of memory, and answers in
under a second on a laptop CPU.

The smaller builds are published for cases where download size genuinely
dominates. For anything where a missed detection is a compliance problem, the
extra 232 MB is cheap.

Models: [jeffasante/cellm-models](https://huggingface.co/jeffasante/cellm-models/tree/main/privacy)
