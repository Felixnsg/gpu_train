import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from facenet_pytorch import InceptionResnetV1, MTCNN
import pytorch_trainer

# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1,2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Dataset class - focused only on data loading
class Facedataset(Dataset):
    def __init__(self, root_dir, transform=None):        
        super().__init__()
        self.transform = transform 
        self.image_path = [] #--- A empty list of all the image paths we are gonna use
        self.image_label = [] #--- The labes

        felix_dir = os.path.join(root_dir, 'felix') #--- directory for felix pictures
        for img_name in os.listdir(felix_dir):
            if img_name.endswith((".jpg", ".jpeg", ".png")): #--- check if the image is valid
                self.image_path.append(os.path.join(felix_dir, img_name)) #--- To the list path, we add the image if valid.
                self.image_label.append(1) #--- if felix we give label 1

        notfelix_dir = os.path.join(root_dir, "notfelix")
        for img_name in os.listdir(notfelix_dir):
            if img_name.endswith((".jpg", ".jpeg", ".png")):
                self.image_path.append(os.path.join(notfelix_dir, img_name))
                self.image_label.append(0) #--- if not felix, we give label 0

    def __len__(self):
        return len(self.image_path) #--- len of images paths.
    
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_label[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def train_test_split(self):
        X = self.image_path
        y = self.image_label

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=self.image_label)

        return X_train, y_train, X_test, y_test

# Separate class for applying transforms to subsets
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

# Main code (this would go in your main script)
def main(data_path = None):
    # Create dataset
    data_path = r"C:\Users\Felix\Documents\ModelBuilding\data\processed"

    dataset = Facedataset(root_dir=data_path, transform=None)
    
    # Split dataset
    X_train, y_train, X_test, y_test = dataset.train_test_split()
    
    # Create indices for subsets
    train_indices = [i for i in range(len(dataset)) if dataset.image_path[i] in X_train]
    test_indices = [i for i in range(len(dataset)) if dataset.image_path[i] in X_test]
    
    # Create subsets
    train_subset = Subset(dataset=dataset, indices=train_indices)
    test_subset = Subset(dataset=dataset, indices=test_indices)
    
    # Apply transforms
    train_transformed = TransformSubset(train_subset, transform=train_transforms)
    test_transformed = TransformSubset(test_subset, transform=val_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_transformed, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_transformed, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

 
model = InceptionResnetV1(pretrained='vggface2').eval()

# If you have a GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    
for param in model.parameters():
    param.requires_grad = False


class FelixClassifier(torch.nn.Module):
    def __init__(self, model):
        super(FelixClassifier, self).__init__()

        self.model = model
        self.classifier = torch.nn.Linear(512, 2)

    def forward(self, x):
        embedding = self.model(x)
        output = self.classifier(embedding)
        return output

felix_classifier = FelixClassifier(model)

if torch.cuda.is_available():
    felix_classifier = felix_classifier.to('cuda')

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(felix_classifier.classifier.parameters(), lr=0.001)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

def trainMe():
    trainer = pytorch_trainer.PyTorchTrainer(
        model = felix_classifier, 
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler =  scheduler)

    train_loader, test_loader = main()

    history = trainer.train(
        train_loader = train_loader,
        val_loader = test_loader,
        epochs = 70,
        early_stopping = True,
        patience = 35 ,
        save_path = 'models/felix_classifier.pth',
        save_best_only =True,
        verbose = 1

    )


    trainer.get_classification_metrics(
        data_loader=test_loader,
        class_names = ["Not Felix", "Felix"]
    )

    trainer.visualize_predictions(
        data_loader = test_loader,
        class_names = ['not Felix', 'Felix'],
        num_images = 30
    )



def fine_tune_model(fine_tune_path, checkpoint_path='models/felix_classifier.pth', epochs=20):
    # Load your model architecture
    base_model = InceptionResnetV1(pretrained='vggface2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        base_model = base_model.to(device)
    
    # Create The new classifier, same architecture.
    model = FelixClassifier(base_model)
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Load the pre-trained weights
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get data loaders for your new data
    train_loader, test_loader = main(data_path = fine_tune_path)
    
    # Freeze early layers, unfreeze later layers
    for name, param in model.model.named_parameters():
        if "block8" in name or "last_linear" in name or "logits" in name:
            param.requires_grad = True  # Fine-tune these layers
        else:
            param.requires_grad = False  # Keep these frozen
    
    # Set up optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.model.named_parameters()  #--- Parameters for the just unfrozen layers
                  if p.requires_grad], 'lr': 1e-5},
        {"params": model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    # Set up scheduler and loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = pytorch_trainer.PyTorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Train with early stopping
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        early_stopping=True,
        patience=10,
        save_path='models/finetuned_felix_classifier.pth',
        save_best_only=True,
        verbose=1
    )
    
    trainer.plot_training_history()

    # Evaluate and visualize
    trainer.get_classification_metrics(
        data_loader=test_loader,
        class_names=["Not Felix", "Felix"]
    )
    
    trainer.visualize_predictions(
        data_loader=test_loader,
        class_names=['not Felix', 'Felix'],
        num_images=30
    )
    
    return model, history


model, history = fine_tune_model(fine_tune_path= r'C:\Users\Felix\Desktop\Saved_Images', checkpoint_path = r"C:\Users\Felix\Desktop\felix_classifier.pth",
                                 epochs=50)