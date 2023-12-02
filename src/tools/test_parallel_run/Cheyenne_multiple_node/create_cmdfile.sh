rm testcmdfile
for i in {1..360}
do
echo "echo start $i; sleep 1; echo finish $i" >> testcmdfile
done