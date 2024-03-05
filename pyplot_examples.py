import random
import matplotlib.pyplot as plt


num = random.randrange(0, 100)
print(f"num = {num}")


plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
plt.show()

plt.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
plt.show()
