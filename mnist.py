#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflowjs as tfjs

# create the model
def createModel(hidden_layer_size): 
	output_size = 10
	    
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
		tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
		tf.keras.layers.Dense(hidden_layer_size, activation='tanh'), # 2nd hidden layer
		tf.keras.layers.Dense(hidden_layer_size, activation='tanh'), # 3rd hidden layer    
		tf.keras.layers.Dense(output_size, activation='softmax') # output layer
	])

	model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model

# train the model
def trainModel(model, BUFFER_SIZE, BATCH_SIZE, NUM_EPOCHS):
	# data
	mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
	mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

	num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
	num_validation_samples = tf.cast(num_validation_samples, tf.int64)
	num_test_samples = mnist_info.splits['test'].num_examples
	num_test_samples = tf.cast(num_test_samples, tf.int64)

	shuffled_train_and_validation_data = mnist_train.shuffle(BUFFER_SIZE)
	validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
	train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
	
	train_data = train_data.batch(BATCH_SIZE)
	validation_data = validation_data.batch(num_validation_samples)
	test_data = mnist_test.batch(num_test_samples)
	validation_inputs, validation_targets = next(iter(validation_data))
	
	# training
	model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =2, validation_steps=10)
	
	# run model on test data
	test_loss, test_accuracy = model.evaluate(test_data)
	print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

# save for tfjs
def saveModel(model):		
	# save
	tf.saved_model.save(model, "mnist")
	# save for tfjs
	tfjs.converters.save_keras_model(model, "mnistjs")
	# save for tflite
	tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
	open("mnist.tflite", "wb").write(tflite_model)
	
if __name__ == '__main__':
	hidden_layer_size = 200
	model=createModel(hidden_layer_size)
	model.summary()
	BUFFER_SIZE = 10000
	BATCH_SIZE = 100
	NUM_EPOCHS = 15
	trainModel(model, BUFFER_SIZE, BATCH_SIZE, NUM_EPOCHS)
	saveModel(model)
