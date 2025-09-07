import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
#comment
class PestDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        Initialize the pest detection model.
        
        Args:
            num_classes (int): Number of classes (including background)
        """
        self.num_classes = num_classes
        
        # Load a pre-trained model for feature extraction
        backbone = torchvision.models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Define the number of output channels for the backbone
        self.out_channels = 2048  # For ResNet50
        
        # Define the RPN (Region Proposal Network)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define the RoI (Region of Interest) pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            # Use a dummy backbone that we'll replace
            backbone=torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=800,
            max_size=1333
        )
        
        # Replace the backbone with our feature extractor
        self.model.backbone = self.backbone
        
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images (list[Tensor]): List of input images
            targets (list[dict], optional): List of target dictionaries
            
        Returns:
            If training, returns a dictionary of losses.
            If inference, returns a list of dictionaries containing detections.
        """
        return self.model(images, targets)
    
    def save(self, path):
        """Save the model weights to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(num_classes=checkpoint['num_classes'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


def create_model(num_classes, pretrained=True):
    """
    Create a Faster R-CNN model with a ResNet-50 backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): If True, use pre-trained weights
        
    Returns:
        A Faster R-CNN model
    """
    # Load a pre-trained model for feature extraction
    backbone = torchvision.models.resnet50(pretrained=pretrained)
    # Remove the last fully connected layer and average pooling
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    
    # Freeze the first few layers
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in list(backbone.parameters())[-10:]:
        param.requires_grad = True
    
    # Define the RPN (Region Proposal Network)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Define the RoI (Region of Interest) pooling
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the Faster R-CNN model
    model = FasterRCNN(
        # Use a dummy backbone that we'll replace
        backbone=torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333
    )
    
    # Replace the backbone with our feature extractor
    model.backbone = backbone
    
    return model
