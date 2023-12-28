import matplotlib.pyplot as plt
all_the_text = open('./log_sgd_big_32.txt').read()
history = type(eval(all_the_text))
print history
# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
