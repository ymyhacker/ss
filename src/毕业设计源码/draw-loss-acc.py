#import pylab as plt
import matplotlib.pyplot as plt
import sys
strs = sys.argv[1]
with open(strs) as r:
    acc = r.read()
print type(acc)
history = eval(acc)
print type(history)
# print history.get('acc')

history = eval(acc)
plt.plot(history.get('acc'))
plt.plot(history.get('val_acc'))
plt.plot(history.get('loss'))
plt.plot(history.get('val_loss'))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
