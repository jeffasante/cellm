require 'json'
file = ARGV[0]
File.open(file, 'rb') do |f|
  magic = f.read(5)
  ver = f.read(1).unpack1('C')
  len = f.read(4).unpack1('V')
  puts JSON.pretty_generate(JSON.parse(f.read(len)))
end
