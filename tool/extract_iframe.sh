#!/bin/sh

lookup_args() {
    local ret=0

    while [ "$#" -ne 0 ]; do
        if [ "$1" = "-i" ]; then
            shift
            input_dir="$1"
            if [ ! -d "$input_dir" ]; then
                ret=1
                echo "Error: $input_dir is not a directory"
            fi
        elif [ "$1" = "-o" ]; then
            shift
            output_dir="$1"
            if [ ! -d "$output_dir" ]; then
                mkdir -p $output_dir

                if [ -d "$output_dir" ]; then
                    echo "Info: $output_dir was made"
                else
                    echo "Error: couldn't make $output_dir"
                    ret=1
                fi
            fi
        fi

        shift
    done

    return $ret
}

check_args() {
    local ret=0

    if [ "$#" -ne 4 ]; then
        echo "Error: wrong arguments"
        ret=1
    else
        lookup_args "$@"
        ret=$?
    fi

    return $ret
}

print_usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR"
}

## MAIN SCRIPT ##

input_dir=""
output_dir=""

check_args "$@"
if [ "$?" -ne 0 ]; then
    print_usage
    exit 1
fi

echo "Input_dir: $input_dir"
echo "Output_dir: $output_dir"

find $input_dir -type f -name "*.mp4" | while read path; do
    echo "Processing: $path: "
    filename=$(basename -- "$path")
    purename="${filename%.*}"

    ffmpeg -nostdin -loglevel error -i $path -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr $output_dir/${purename}_%04d.png
done

echo "Done"

