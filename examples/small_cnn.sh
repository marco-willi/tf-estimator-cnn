python main.py \
--root_path D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all\\ \
--model_save_path ./data/model_run \
--model small_cnn \
--max_epoch 10 \
--batch_size 64 \
--image_size 50 \
--num_gpus 0 \
--num_cpus 2 \
--train_fraction 0.8 \
--color_augmentation True \
--weight_decay 0.001