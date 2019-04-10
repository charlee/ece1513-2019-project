# 1.3
echo "Running a1-1.3..."
rm -rf a1-1.3
python a1.py --alg lr --path a1-1.3 --alpha 0.005 0.001 0.0001 --epochs 5000
echo "done."

# # 1.4
echo "Running a1-1.4..."
rm -rf a1-1.4
python a1.py --alg lr --path a1-1.4 --alpha 0.005 --lambda 0.001 0.1 0.5 --epochs 5000
echo "done."

# # 1.5
echo "Running a1-1.5..."
rm -rf a1-1.5
python a1.py --alg lrne --path a1-1.5
echo "done."

# # 2.2
echo "Running a1-2.2..."
rm -rf a1-2.2
python a1.py --alg=log --alpha=0.005 --lambda=0.1 --epochs=5000 --path=a1-2.2
echo "done."

echo "Running a1-2.3..."
rm -rf a1-2.3
python a1.py --alg=log --alpha=0.005 --lambda=0 --epochs=5000 --path=a1-2.3
echo "done."

# # 3.2
echo "Running a1-3.2..."
rm -rf a1-3.2
python a1-tf.py --path=a1-3.2 --optimizer=adam --loss=mse --epochs=700 --batch_size=500 --lambda 0 --alpha 0.001
echo "done."

# 3.3
echo "Running a1-3.3..."
rm -rf a1-3.3
python a1-tf.py --path=a1-3.3 --optimizer=adam --loss=mse --epochs=700 --batch_size 100 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.3 --optimizer=adam --loss=mse --epochs=700 --batch_size 700 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.3 --optimizer=adam --loss=mse --epochs=700 --batch_size 1750 --lambda 0 --alpha 0.001
echo "done."

# 3.4
echo "Running a1-3.4..."
rm -rf a1-3.4
python a1-tf.py --path=a1-3.4 --optimizer=adam --loss=mse --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --beta1 0.95 0.99
python a1-tf.py --path=a1-3.4 --optimizer=adam --loss=mse --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --beta2 0.99 0.9999
python a1-tf.py --path=a1-3.4 --optimizer=adam --loss=mse --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --epsilon 1e-9 1e-4
echo "done."

# 3.5
echo "Running a1-3.5..."
rm -rf a1-3.5
python a1-tf.py --path=a1-3.5 --optimizer=gd --loss=ce --epochs=700 --batch_size=500 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 100 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 700 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 1750 --lambda 0 --alpha 0.001
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --beta1 0.95 0.99
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --beta2 0.99 0.9999
python a1-tf.py --path=a1-3.5 --optimizer=adam --loss=ce --epochs=700 --batch_size 500 --lambda 0 --alpha 0.001 --epsilon 1e-9 1e-4
echo "done."