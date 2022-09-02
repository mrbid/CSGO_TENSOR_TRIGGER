function renumber_files
{
	for i in *.ppm; do
		mv -i -- "$i" "$RANDOM$RANDOM$RANDOM.ppm"
	done
	a=1
	for i in *.ppm; do
		new=$(printf "%d.ppm" "$a")
		mv -i -- "$i" "$new"
		let a=a+1
	done
}
cd target
renumber_files
echo renumbered targets
cd ../nontarget
renumber_files
echo renumbered nontargets