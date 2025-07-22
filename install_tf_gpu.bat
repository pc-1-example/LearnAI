@echo off
echo Удаляем старые версии TensorFlow...
pip uninstall -y tensorflow tensorflow-intel

echo.
echo Устанавливаем TensorFlow 2.13.0 с поддержкой GPU...
pip install tensorflow==2.13.0

echo.
echo Создаем файл для проверки GPU...
echo import tensorflow as tf > check_tf_gpu.py
echo print("TF version:", tf.__version__) >> check_tf_gpu.py
echo print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU'))) >> check_tf_gpu.py
echo print("Physical devices:", tf.config.list_physical_devices()) >> check_tf_gpu.py

echo.
echo Запускаем проверку TensorFlow...
python check_tf_gpu.py

pause
