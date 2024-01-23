python3 ../llama.cpp/convert.py --outfile tinymedical.gguf Model
../llama.cpp/main -ngl 32 -m tinymedical.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -i -ins