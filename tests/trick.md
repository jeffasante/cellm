1.
./target/release/infer \
  --model models/LFM2.5-230M.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt "A farmer has a wolf, a goat, and a cabbage. He needs to cross a river with them, but his boat can only carry him and one item at a time. If left alone, the wolf eats the goat, and the goat eats the cabbage. How can he get everything across safely? Walk through the solution step by step." \
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu \
  2>&1


./target/release/infer \
  --model models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt "A farmer has a wolf, a goat, and a cabbage. He needs to cross a river with them, but his boat can only carry him and one item at a time. If left alone, the wolf eats the goat, and the goat eats the cabbage. How can he get everything across safely? Walk through the solution step by step." \
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu \
  2>&1



./target/release/infer \
  --model /Users/jeff/Desktop/cellm/models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt 'A grocery sells a bag of ice for $1.25, and makes 20% profit. If it sells 500 bags of ice, how much total profit does it make? A)125", "B)150", "C)225", "D)250", "E)275' \
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu


./target/release/infer \
  --model /Users/jeff/Desktop/cellm/models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4-v2.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt 'The original price of an item is discounted 22%. A customer buys the item at this discounted price using a $20-off coupon. There is no tax on the item, and this was the only item the customer bought. If the customer paid $1.90 more than half the original price of the item, what was the original price of the item? "A)$61", "B)$65", "C)$67.40", "D)$70", "E)$78.20". Also add the rationale' \
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu



./target/release/infer \
  --model /Users/jeff/Desktop/cellm/models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt 'The original price of an item is discounted 22%. A customer buys the item at this discounted price using a $20-off coupon. There is no tax on the item, and this was the only item the customer bought. If the customer paid $1.90 more than half the original price of the item, what was the original price of the item? "A)$61", "B)$65", "C)$67.40", "D)$70", "E)$78.20". Also add the rationale' \
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu




'

./target/release/infer \
  --model /Users/jeff/Desktop/cellm/models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4.cellm \
  --tokenizer models/LFM2.5-230M/tokenizer.json \
  --prompt "
Meeting Transcript

Date: June 24, 2026

Attendees

* Jeff
* Sarah (Product Manager)
* Michael (Engineering)
* Anita (Finance)
* David (Infrastructure)

⸻

Sarah:
Let’s go through the release blockers before next week’s deployment.

Michael:
Authentication is finished, but we’re still waiting on Infrastructure to provision the production Redis cluster.

David:
I’ll have that ready by Thursday afternoon.

Jeff:
Perfect. Once Redis is available, I’ll verify the session handling and update the deployment documentation.

Sarah:
Jeff, can you also send me the architecture diagram you showed yesterday? I want to include it in tomorrow’s stakeholder presentation.

Jeff:
Sure, I’ll send it after this meeting.

Anita:
Finance still hasn’t received the cloud cost estimate for the observability platform.

Jeff:
That’s on me. I’ll prepare the estimate and email it before Friday.

Michael:
We also need API rate limits documented.

Jeff:
Good point. I’ll create a Confluence page for that.

Sarah:
Did Legal ever approve the vendor agreement?

Jeff:
Not yet. I said I’d check with Legal this week.

David:
When you hear back, let us know because procurement is waiting.

Jeff:
Will do.

Michael:
Can someone remind me about the Grafana dashboard permissions?

Jeff:
I’ll send you the RBAC documentation after we’re done here.

Sarah:
The observability demo went well. We should schedule another session with the ERP team.

Jeff:
I’ll reach out to them and find a date.

Anita:
And don’t forget to introduce me to the procurement manager.

Jeff:
Right. I’ll make that introduction tomorrow.


Who have I promised to follow up with?
"\
  --chat \
  --chat-format auto \
  --gen 256 \
  --temperature 0.1 \
  --top-k 50 \
  --repeat-penalty 1.05 \
  --stop-eos \
  --backend cpu
  
'
---------------------------------------------


./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "
Meeting Transcript

Date: June 24, 2026

Attendees

* Jeff
* Sarah (Product Manager)
* Michael (Engineering)
* Anita (Finance)
* David (Infrastructure)

⸻

Sarah:
Let’s go through the release blockers before next week’s deployment.

Michael:
Authentication is finished, but we’re still waiting on Infrastructure to provision the production Redis cluster.

David:
I’ll have that ready by Thursday afternoon.

Jeff:
Perfect. Once Redis is available, I’ll verify the session handling and update the deployment documentation.

Sarah:
Jeff, can you also send me the architecture diagram you showed yesterday? I want to include it in tomorrow’s stakeholder presentation.

Jeff:
Sure, I’ll send it after this meeting.

Anita:
Finance still hasn’t received the cloud cost estimate for the observability platform.

Jeff:
That’s on me. I’ll prepare the estimate and email it before Friday.

Michael:
We also need API rate limits documented.

Jeff:
Good point. I’ll create a Confluence page for that.

Sarah:
Did Legal ever approve the vendor agreement?

Jeff:
Not yet. I said I’d check with Legal this week.

David:
When you hear back, let us know because procurement is waiting.

Jeff:
Will do.

Michael:
Can someone remind me about the Grafana dashboard permissions?

Jeff:
I’ll send you the RBAC documentation after we’re done here.

Sarah:
The observability demo went well. We should schedule another session with the ERP team.

Jeff:
I’ll reach out to them and find a date.

Anita:
And don’t forget to introduce me to the procurement manager.

Jeff:
Right. I’ll make that introduction tomorrow.


Who have I promised to follow up with?
"\
  --chat --gen 300 --temperature 0 --backend metal --kv-encoding f16


  

---------------------------------------------

  ---
  cd /Users/jeff/Desktop/cellm && ./target/release/infer \
    --model models/LFM2.5-230M-int4-v2.cellm \
    --tokenizer models/LFM2.5-230M/tokenizer.json \
    --prompt 'Extract JSON from this CV:
  "Jeff Asante
  Senior Software Engineer
  Email: jeff@ecg.com.gh | Phone: +233 50 123 4567
  
  SUMMARY
  Experienced software engineer with 8+ years building distributed systems.
  
  EXPERIENCE
  Electricity Company of Ghana, Accra — Senior Software Engineer
  2020 — Present
  - Built the resume screening AI platform using FastAPI and LLMs
  - Deployed monitoring infrastructure with Prometheus and Grafana
  - Managed microservices on Docker Swarm
  
  TechCorp, Accra — Software Engineer
  2016 — 2020
  - Developed REST APIs in Python and Go
  - Led migration from monolith to microservices
  
  EDUCATION
  University of Ghana — BSc Computer Science (2012 — 2016)"
  
  Keys: name, title, email, phone, summary, experience, education' \
    --chat --chat-format auto --gen 256 --temperature 0 --stop-eos --backend cpu 2>&1 | tail -30