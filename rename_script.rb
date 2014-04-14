#Rename SCI filename to healthy filename
puts "Renaming files..."

Dir.glob("*").sort.each do |f|
	filename = File.basename(f, File.extname(f))
	if filename.match(/SCI/)
	  filename.gsub!(/SCI/,'healthy')
	  puts filename
	  File.rename(f, filename + File.extname(f))
	end
end

puts "Renaming complete."
