{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a895c021",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid non-printable character U+00A0 (628955864.py, line 7)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[1], line 7\u001b[1;36m\u001b[0m\n\u001b[1;33m    from torch_geometric.data import Data\u001b[0m\n\u001b[1;37m                             ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid non-printable character U+00A0\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import sklearn.metrics as metrics\n",
    "from torch_geometric.data import Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "78ae4a26",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid non-printable character U+00A0 (2356146044.py, line 14)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[2], line 14\u001b[1;36m\u001b[0m\n\u001b[1;33m    shuffle=False, num_workers=2)\u001b[0m\n\u001b[1;37m                  ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid non-printable character U+00A0\n"
     ]
    }
   ],
   "source": [
    "BATCH_SIZE = 32\n",
    "\n",
    "# Définir les transformations pour les images\n",
    "transform = transforms.Compose([transforms.ToTensor()])\n",
    "\n",
    "# Télécharger l'ensemble de données MNSIT et le charger\n",
    "trainset = torchvision.datasets.MNIST(root='C:/Users/marya/OneDrive/Bureau/pfe/mnist/MNIST', train=True,\n",
    "                                        download=True, transform=transform)\n",
    "trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,\n",
    "                                          shuffle=True, num_workers=2)\n",
    "testset = torchvision.datasets.MNIST(root='C:/Users/marya/OneDrive/Bureau/pfe/mnist/MNIST', train=False,\n",
    "                                       download=True, transform=transform)\n",
    "testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,\n",
    "                                         shuffle=False, num_workers=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "71f1ff34",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'trainset' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[3], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;28mlen\u001b[39m(\u001b[43mtrainset\u001b[49m))\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28mprint\u001b[39m(trainset[\u001b[38;5;241m10\u001b[39m])\n",
      "\u001b[1;31mNameError\u001b[0m: name 'trainset' is not defined"
     ]
    }
   ],
   "source": [
    "print(len(trainset))\n",
    "print(trainset[10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8ae3ca02",
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unindent does not match any outer indentation level (<tokenize>, line 26)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m<tokenize>:26\u001b[1;36m\u001b[0m\n\u001b[1;33m    return out\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mIndentationError\u001b[0m\u001b[1;31m:\u001b[0m unindent does not match any outer indentation level\n"
     ]
    }
   ],
   "source": [
    "class MyModel(nn.Module):\n",
    "    def _init_(self):\n",
    "        super(MyModel, self)._init_()\n",
    "\n",
    "        # 28x28x1 => 26x26x32\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)\n",
    "        self.d1 = nn.Linear(26 * 26 * 32, 128)\n",
    "        self.d2 = nn.Linear(128, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # 32x1x28x28 => 32x32x26x26\n",
    "        x = self.conv1(x)\n",
    "        x = F.relu(x)\n",
    "\n",
    "        # flatten => 32 x (32*26*26)\n",
    "        x = x.flatten(start_dim = 1)\n",
    "        #x = x.view(32, -1)\n",
    "\n",
    "        # 32 x (32*26*26) => 32x128\n",
    "        x = self.d1(x)\n",
    "        x = F.relu(x)\n",
    "\n",
    "        # logits => 32x10\n",
    "        logits = self.d2(x)\n",
    "        out = F.softmax(logits, dim=1)\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "71970d5b",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'torch' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m device \u001b[38;5;241m=\u001b[39m \u001b[43mtorch\u001b[49m\u001b[38;5;241m.\u001b[39mdevice(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcuda:0\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mcuda\u001b[38;5;241m.\u001b[39mis_available() \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcpu\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      2\u001b[0m model \u001b[38;5;241m=\u001b[39m MyModel()\n\u001b[0;32m      3\u001b[0m model \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mto(device)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
     ]
    }
   ],
   "source": [
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = MyModel()\n",
    "model = model.to(device)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2c55c4cf",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'torch' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[7], line 6\u001b[0m\n\u001b[0;32m      3\u001b[0m learning_rate \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0.001\u001b[39m\n\u001b[0;32m      4\u001b[0m num_epochs \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m5\u001b[39m\n\u001b[1;32m----> 6\u001b[0m device \u001b[38;5;241m=\u001b[39m \u001b[43mtorch\u001b[49m\u001b[38;5;241m.\u001b[39mdevice(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcuda:0\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mcuda\u001b[38;5;241m.\u001b[39mis_available() \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcpu\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      7\u001b[0m model \u001b[38;5;241m=\u001b[39m MyModel()\n\u001b[0;32m      8\u001b[0m model \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mto(device)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "learning_rate = 0.001\n",
    "num_epochs = 5\n",
    "\n",
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = MyModel()\n",
    "model = model.to(device)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)\n",
    "\n",
    "train_loss_history = []\n",
    "train_acc_history = []\n",
    "test_acc_history = []\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    train_running_loss = 0.0\n",
    "    train_acc = 0.0\n",
    "\n",
    "    ## training step\n",
    "    for i, (images, labels) in enumerate(trainloader):\n",
    "        \n",
    "        images = images.to(device)\n",
    "        labels = labels.to(device)\n",
    "\n",
    "        ## forward + backprop + loss\n",
    "        logits = model(images)\n",
    "        loss = criterion(logits, labels)\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "\n",
    "        ## update model params\n",
    "        optimizer.step()\n",
    "\n",
    "        train_running_loss += loss.detach().item()\n",
    "        train_acc += (torch.argmax(logits, 1).flatten() == labels).type(torch.float).mean().item()\n",
    "    \n",
    "    train_loss_history.append(train_running_loss / i)\n",
    "    train_acc_history.append(train_acc/i * 100)\n",
    "\n",
    "    print('Epoch: %d | Loss: %.4f |  Train Accuracy: %.2f%%' \\\n",
    "          %(epoch,train_loss_history[-1],train_acc_history[-1]))\n",
    "\n",
    "      \n",
    "    test_acc = 0.0\n",
    "    for i, (images, labels) in enumerate(testloader, 0):\n",
    "        images = images.to(device)\n",
    "        labels = labels.to(device)\n",
    "        outputs = model(images)\n",
    "        test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item()\n",
    "    test_acc_history.append(test_acc/i * 100)\n",
    "\n",
    "    print('Test Accuracy: %.2f%%' %(test_acc_history[-1]))\n",
    "\n",
    "\n",
    "# plot the graphs\n",
    "plt.plot(train_loss_history)\n",
    "plt.title(\"Training Loss\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.show()\n",
    "\n",
    "plt.plot(train_acc_history, label=\"Training Accuracy\")\n",
    "plt.plot(test_acc_history, label=\"Test Accuracy\")\n",
    "plt.title(\"Accuracy\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "24e6f57b",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid non-printable character U+00A0 (3552148226.py, line 10)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[8], line 10\u001b[1;36m\u001b[0m\n\u001b[1;33m    print('Test Accuracy: %.2f%%' % accuracy)\u001b[0m\n\u001b[1;37m                                 ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid non-printable character U+00A0\n"
     ]
    }
   ],
   "source": [
    "test_acc = 0.0\n",
    "for i, (images, labels) in enumerate(testloader, 0):\n",
    "    images = images.to(device)\n",
    "    labels = labels.to(device)\n",
    "    outputs = model(images)\n",
    "    test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item()\n",
    "    preds = torch.argmax(outputs, 1).flatten().cpu().numpy()\n",
    "\n",
    "accuracy = 100 * test_acc / i\n",
    "print('Test Accuracy: %.2f%%' % accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d542ab02",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'testloader' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[9], line 19\u001b[0m\n\u001b[0;32m     16\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m predicted\u001b[38;5;241m.\u001b[39mitem()\n\u001b[0;32m     18\u001b[0m \u001b[38;5;66;03m# get a batch of test data\u001b[39;00m\n\u001b[1;32m---> 19\u001b[0m images, labels \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mnext\u001b[39m(\u001b[38;5;28miter\u001b[39m(\u001b[43mtestloader\u001b[49m))\n\u001b[0;32m     21\u001b[0m \u001b[38;5;66;03m# select a random image from the batch\u001b[39;00m\n\u001b[0;32m     22\u001b[0m idx \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mrandint(\u001b[38;5;241m0\u001b[39m, images\u001b[38;5;241m.\u001b[39msize()[\u001b[38;5;241m0\u001b[39m], (\u001b[38;5;241m1\u001b[39m,))\u001b[38;5;241m.\u001b[39mitem()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'testloader' is not defined"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def predict_image(img, model):\n",
    "    # make sure the model is in evaluation mode\n",
    "    model.eval()\n",
    "\n",
    "    # move image tensor to device\n",
    "    img = img.to(device)\n",
    "\n",
    "    # forward pass through the model\n",
    "    with torch.no_grad():\n",
    "        outputs = model(img.unsqueeze(0))\n",
    "        _, predicted = torch.max(outputs.data, 1)\n",
    "        \n",
    "    return predicted.item()\n",
    "\n",
    "# get a batch of test data\n",
    "images, labels = next(iter(testloader))\n",
    "\n",
    "# select a random image from the batch\n",
    "idx = torch.randint(0, images.size()[0], (1,)).item()\n",
    "\n",
    "# get the corresponding label and image\n",
    "label = labels[idx].item()\n",
    "img = images[idx]\n",
    "\n",
    "# predict the label of the image\n",
    "predicted_label = predict_image(img, model)\n",
    "\n",
    "# show the image\n",
    "plt.imshow(img.permute(1, 2, 0))\n",
    "plt.axis('off')\n",
    "plt.show()\n",
    "\n",
    "# print the true label and predicted label\n",
    "print('True Label:', label)\n",
    "print('Predicted Label:', predicted_label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b5c907ef",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'testloader' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[10], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# get a batch of test data\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m images, labels \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mnext\u001b[39m(\u001b[38;5;28miter\u001b[39m(\u001b[43mtestloader\u001b[49m))\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# select a random image from the batch\u001b[39;00m\n\u001b[0;32m      5\u001b[0m idx \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mrandint(\u001b[38;5;241m0\u001b[39m, images\u001b[38;5;241m.\u001b[39msize()[\u001b[38;5;241m0\u001b[39m], (\u001b[38;5;241m1\u001b[39m,))\u001b[38;5;241m.\u001b[39mitem()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'testloader' is not defined"
     ]
    }
   ],
   "source": [
    "# get a batch of test data\n",
    "images, labels = next(iter(testloader))\n",
    "\n",
    "# select a random image from the batch\n",
    "idx = torch.randint(0, images.size()[0], (1,)).item()\n",
    "\n",
    "# get the corresponding label and image\n",
    "label = labels[idx].item()\n",
    "img = images[idx]\n",
    "\n",
    "# predict the label of the image\n",
    "predicted_label = predict_image(img, model)\n",
    "\n",
    "# show the image\n",
    "plt.imshow(img.permute(1, 2, 0))\n",
    "plt.axis('off')\n",
    "plt.show()\n",
    "\n",
    "# print the true label and predicted label\n",
    "print('True Label:', label)\n",
    "print('Predicted Label:', predicted_label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2334a21f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
