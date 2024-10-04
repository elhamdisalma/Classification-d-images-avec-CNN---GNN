{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "17bcc282",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages.\n",
    "import os\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import torch_geometric\n",
    "from torch_geometric.data import DataLoader\n",
    "import torch.optim as optim\n",
    "\n",
    "\n",
    "torch.multiprocessing.set_sharing_strategy('file_system')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "93c8140b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Définir les transformations pour les images\n",
    "transform = transforms.Compose([\n",
    "    transforms.Grayscale(num_output_channels=1),\n",
    "    transforms.Resize((64, 64)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5,), (0.5,))\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4c2e7683",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files already downloaded and verified\n",
      "Files already downloaded and verified\n",
      "50000\n",
      "(tensor([[[-0.5294, -0.5294, -0.5373,  ..., -0.6863, -0.7569, -0.7882],\n",
      "         [-0.5451, -0.5373, -0.5451,  ..., -0.6863, -0.7490, -0.7804],\n",
      "         [-0.5686, -0.5608, -0.5529,  ..., -0.6941, -0.7412, -0.7647],\n",
      "         ...,\n",
      "         [-0.3804, -0.3725, -0.3647,  ..., -0.5294, -0.5922, -0.6235],\n",
      "         [-0.3647, -0.3647, -0.3569,  ..., -0.5294, -0.5843, -0.6157],\n",
      "         [-0.3569, -0.3569, -0.3490,  ..., -0.5294, -0.5843, -0.6157]]]), 4)\n"
     ]
    }
   ],
   "source": [
    "# Télécharger l'ensemble de données CIFAR-10 et le charger\n",
    "trainset = torchvision.datasets.CIFAR10(root='C:/Users/hp/Desktop/Python Projects/cifar-10-batches-py', train=True,\n",
    "                                        download=True, transform=transform)\n",
    "testset = torchvision.datasets.CIFAR10(root='C:/Users/hp/Desktop/Python Projects/cifar-10-batches-py', train=False,\n",
    "                                       download=True, transform=transform)\n",
    "trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,\n",
    "                                          shuffle=True, num_workers=2)\n",
    "testloader = torch.utils.data.DataLoader(testset, batch_size=64,\n",
    "                                         shuffle=False, num_workers=2)\n",
    "\n",
    "\n",
    "# Définir les classes de l'ensemble de données CIFAR-10\n",
    "classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')\n",
    "print(len(trainset))\n",
    "print(trainset[10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0dc1d467",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torch._six'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[4], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnn\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgcn_lib\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdense\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mtorch_vertex\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m DynConv2d\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01m_six\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m inf\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# gcn_lib is downloaded from https://github.com/lightaime/deep_gcns_torch\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\gcn_lib\\__init__.py:5\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mtorch_nn\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mtorch_edge\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m----> 5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mtorch_vertex\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\gcn_lib\\torch_vertex.py:10\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpos_embed\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m get_2d_relative_pos_embed\n\u001b[0;32m      9\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mfunctional\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mF\u001b[39;00m\n\u001b[1;32m---> 10\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtimm\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlayers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m DropPath\n\u001b[0;32m     13\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m \u001b[38;5;21;01mMRConv2d\u001b[39;00m(nn\u001b[38;5;241m.\u001b[39mModule):\n\u001b[0;32m     14\u001b[0m     \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m     15\u001b[0m \u001b[38;5;124;03m    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type\u001b[39;00m\n\u001b[0;32m     16\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\__init__.py:2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mversion\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m __version__\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m create_model, list_models, is_model, list_modules, model_entrypoint, \\\n\u001b[0;32m      3\u001b[0m     is_scriptable, is_exportable, set_scriptable, set_exportable\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\__init__.py:1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcspnet\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdensenet\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdla\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\cspnet.py:20\u001b[0m\n\u001b[0;32m     17\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mfunctional\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mF\u001b[39;00m\n\u001b[0;32m     19\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtimm\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdata\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD\n\u001b[1;32m---> 20\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mhelpers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m build_model_with_cfg\n\u001b[0;32m     21\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlayers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ClassifierHead, ConvBnAct, DropPath, create_attn, get_norm_act_layer\n\u001b[0;32m     22\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mregistry\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m register_model\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\helpers.py:17\u001b[0m\n\u001b[0;32m     14\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mutils\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodel_zoo\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mmodel_zoo\u001b[39;00m\n\u001b[0;32m     16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mfeatures\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m FeatureListNet, FeatureDictNet, FeatureHookNet\n\u001b[1;32m---> 17\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlayers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Conv2dSame, Linear\n\u001b[0;32m     20\u001b[0m _logger \u001b[38;5;241m=\u001b[39m logging\u001b[38;5;241m.\u001b[39mgetLogger(\u001b[38;5;18m__name__\u001b[39m)\n\u001b[0;32m     23\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_state_dict\u001b[39m(checkpoint_path, use_ema\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m):\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\layers\\__init__.py:7\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mblur_pool\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m BlurPool2d\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mclassifier\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ClassifierHead, create_classifier\n\u001b[1;32m----> 7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcond_conv2d\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m CondConv2d, get_condconv_initializer\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mconfig\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\\\n\u001b[0;32m      9\u001b[0m     set_layer_config\n\u001b[0;32m     10\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mconv2d_same\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Conv2dSame\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\layers\\cond_conv2d.py:16\u001b[0m\n\u001b[0;32m     13\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m nn \u001b[38;5;28;01mas\u001b[39;00m nn\n\u001b[0;32m     14\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m functional \u001b[38;5;28;01mas\u001b[39;00m F\n\u001b[1;32m---> 16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mhelpers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m to_2tuple\n\u001b[0;32m     17\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mconv2d_same\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m conv2d_same\n\u001b[0;32m     18\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpadding\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m get_padding_value\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\timm\\models\\layers\\helpers.py:6\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;124;03m\"\"\" Layer/Module Helpers\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \n\u001b[0;32m      3\u001b[0m \u001b[38;5;124;03mHacked together by / Copyright 2020 Ross Wightman\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mitertools\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m repeat\n\u001b[1;32m----> 6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01m_six\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m container_abcs\n\u001b[0;32m      9\u001b[0m \u001b[38;5;66;03m# From PyTorch internals\u001b[39;00m\n\u001b[0;32m     10\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_ntuple\u001b[39m(n):\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'torch._six'"
     ]
    }
   ],
   "source": [
    "import torch.nn as nn\n",
    "from gcn_lib.dense.torch_vertex import DynConv2d\n",
    "from torch._six import inf\n",
    "# gcn_lib is downloaded from https://github.com/lightaime/deep_gcns_torch\n",
    "\n",
    "class GrapherModule(nn.Module):\n",
    "    \"\"\"Grapher module with graph conv and FC layers\n",
    "    \"\"\"\n",
    "    def _init_(self, in_channels, hidden_channels, k=9, dilation=1, drop_path=0.0):\n",
    "        super(GrapherModule, self)._init_()\n",
    "        self.fc1 = nn.Sequential(\n",
    "            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),\n",
    "            nn.BatchNorm2d(in_channels),\n",
    "        )\n",
    "        self.graph_conv = nn.Sequential(\n",
    "            DynConv2d(in_channels, hidden_channels, k, dilation, act=None),\n",
    "            nn.BatchNorm2d(hidden_channels),\n",
    "            nn.GELU(),\n",
    "        )\n",
    "        self.fc2 = nn.Sequential(\n",
    "            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),\n",
    "            nn.BatchNorm2d(in_channels),\n",
    "        )\n",
    "        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()\n",
    "\n",
    "    def forward(self, x):\n",
    "        B, C, H, W = x.shape\n",
    "        x = x.reshape(B, C, -1, 1).contiguous()\n",
    "        shortcut = x\n",
    "        x = self.fc1(x)\n",
    "        x = self.graph_conv(x)\n",
    "        x = self.fc2(x)\n",
    "        x = self.drop_path(x) + shortcut\n",
    "        return x.reshape(B, C, H, W)\n",
    "\n",
    "class FFNModule(nn.Module):\n",
    "    \"\"\"Feed-forward Network\n",
    "    \"\"\"\n",
    "    def _init_(self, in_channels, hidden_channels, drop_path=0.0):\n",
    "        super(FFNModule, self)._init_()\n",
    "        self.fc1 = nn.Sequential(\n",
    "            nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0),\n",
    "            nn.BatchNorm2d(hidden_channels),\n",
    "            nn.GELU()\n",
    "        )\n",
    "        self.fc2 = nn.Sequential(\n",
    "            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),\n",
    "            nn.BatchNorm2d(in_channels),\n",
    "        )\n",
    "        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()\n",
    "\n",
    "    def forward(self, x):\n",
    "        shortcut = x\n",
    "        x = self.fc1(x)\n",
    "        x = self.fc2(x)\n",
    "        x = self.drop_path(x) + shortcut\n",
    "        return x\n",
    "\n",
    "class ViGBlock(nn.Module):\n",
    "    \"\"\"ViG block with Grapher and FFN modules\"\"\"\n",
    "    def __init__(self, channels, k, dilation, drop_path=0.0):\n",
    "        super(ViGBlock, self).__init__()\n",
    "        self.grapher = GrapherModule(channels, channels * 2, k, dilation, drop_path)\n",
    "        self.ffn = FFNModule(channels, channels * 4, drop_path)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.grapher(x)\n",
    "        x = self.ffn(x)\n",
    "        return x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a1bbfdb0",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[5], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# print the model\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[43mmodel\u001b[49m)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m# définir la fonction de perte\u001b[39;00m\n\u001b[0;32m      4\u001b[0m criterion \u001b[38;5;241m=\u001b[39m nn\u001b[38;5;241m.\u001b[39mCrossEntropyLoss()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'model' is not defined"
     ]
    }
   ],
   "source": [
    "# print the model\n",
    "print(model)\n",
    "# définir la fonction de perte\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "# définir l'optimiseur\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "def train(model, device, trainloader, optimizer, criterion, epoch):\n",
    "    model.train()\n",
    "    train_loss = 0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "\n",
    "    for batch_idx, (data, target) in enumerate(trainloader):\n",
    "        data, target = data.to(device), target.to(device)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        output = model(data)\n",
    "        loss = criterion(output, target)\n",
    "        loss.backward()\n",
    "\n",
    "        optimizer.step()\n",
    "\n",
    "        train_loss += loss.item()\n",
    "        _, predicted = output.max(1)\n",
    "        total += target.size(0)\n",
    "        correct += predicted.eq(target).sum().item()\n",
    "\n",
    "        if batch_idx % 100 == 0:\n",
    "            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} '\n",
    "                  f'({100. * batch_idx / len(trainloader):.0f}%)]\\tLoss: {loss.item():.6f}')\n",
    "\n",
    "    train_loss /= len(train_loader)\n",
    "    train_acc = 100. * correct / total\n",
    "\n",
    "    print(f'Train Epoch: {epoch} Average loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')\n",
    "\n",
    "    return train_loss, train_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "73141adf",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'ViGBlock' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 4\u001b[0m\n\u001b[0;32m      2\u001b[0m device \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mdevice(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcuda:0\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mcuda\u001b[38;5;241m.\u001b[39mis_available() \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcpu\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m# define the model\u001b[39;00m\n\u001b[1;32m----> 4\u001b[0m model \u001b[38;5;241m=\u001b[39m \u001b[43mViGBlock\u001b[49m(in_channels\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m64\u001b[39m, k\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m9\u001b[39m, dilation\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m, drop_path\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.1\u001b[39m)\n\u001b[0;32m      5\u001b[0m model\u001b[38;5;241m.\u001b[39mto(device)\n\u001b[0;32m      7\u001b[0m \u001b[38;5;66;03m# define the loss and optimizer\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'ViGBlock' is not defined"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "    # define the model\n",
    "    model = ViGBlock(in_channels=64, k=9, dilation=1, drop_path=0.1)\n",
    "    model.to(device)\n",
    "\n",
    "    # define the loss and optimizer\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "    # train the model\n",
    "    train(model, device, trainloader, optimizer, criterion, epoch=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71f2b78a",
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
