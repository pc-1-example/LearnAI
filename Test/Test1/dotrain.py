import gpt_2_simple as gpt2
sess = gpt2.start_tf_sess()

gpt2.finetune(
    sess,
    "train.txt",
    model_name="124M",
    steps=1000,           # число шагов обучения
    restore_from="latest",
    run_name="my_run",
    print_every=100,
    sample_every=200,
    save_every=500
)
