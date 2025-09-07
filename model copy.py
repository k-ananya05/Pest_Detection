import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_backbone():
    """
    Create a ResNet-50 FPN backbone for Faster R-CNN.
    """
    # Load a pre-trained ResNet-50 model with FPN
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)
    return backbone

class PestDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        Initialize the pest detection model.
        
        Args:
            num_classes (int): Number of classes (including background)
        """
        self.num_classes = num_classes
        
        # Get the backbone with FPN
        self.backbone = get_backbone()
        
        # Define the RPN (Region Proposal Network)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * 5,  # 5 feature maps in FPN
            aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Same for each feature map
        )
        
        # Define the RoI (Region of Interest) pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # Use all FPN levels
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone=self.backbone,
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
    Create a Faster R-CNN model with a ResNet-50 FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): If True, use pre-trained weights
        
    Returns:
        A Faster R-CNN model
    """
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model
