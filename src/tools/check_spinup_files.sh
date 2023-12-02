for i in {0..626}
do
    # Count the number of files in the directory
    file_count=$(ls level1_${i}_SpinupFiles/ | wc -l)

    # Check if file count is less than 5
    if [ "$file_count" -lt 5 ]; then
        echo '#######'
        echo $i
        echo $file_count
    fi
done



for i in {0..39}
do
    # Count the number of files in the directory
    file_count=$(ls level2_${i}_SpinupFiles/ | wc -l)

    # Check if file count is less than 5
    if [ "$file_count" -lt 5 ]; then
        echo '#######'
        echo $i
        echo $file_count
    fi
done


for i in {0..3}
do
    # Count the number of files in the directory
    file_count=$(ls level3_${i}_SpinupFiles/ | wc -l)

    # Check if file count is less than 5
    if [ "$file_count" -lt 5 ]; then
        echo '#######'
        echo $i
        echo $file_count
    fi
done
