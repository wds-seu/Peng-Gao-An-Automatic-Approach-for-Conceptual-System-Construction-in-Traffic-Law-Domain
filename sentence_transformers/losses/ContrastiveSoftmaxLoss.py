from .ContrastiveLoss import *
from .SoftmaxLoss import *


class ContrastiveSoftmaxLoss(nn.Module):
    """
    Combine contrastive loss and softmax loss

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension
    :param num_labels
    :param distance_metric
    :param margin
    :param size_average
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
                 margin: float = 0.5,
                 size_average: bool = True):
        super(ContrastiveSoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average
        self.classifier = nn.Linear(2 * sentence_embedding_dimension,
                                    num_labels)
        self.bilinear_W = nn.Parameter(torch.randn(num_labels,
                                                   sentence_embedding_dimension,
                                                   sentence_embedding_dimension))

    def forward(self,
                sentence_features: Iterable[Dict[str, Tensor]],
                labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_a, rep_b = reps

        # contrastive loss
        distances = self.distance_metric(rep_a, rep_b)
        contrastive_losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))

        # softmax loss
        vectors_concat = [rep_a, rep_b]
        features = torch.cat(vectors_concat, 1)
        print('softmax input features size:', features.shape)
        output = self.classifier(features)
        print('output size:', output.shape)
        loss_fct = nn.CrossEntropyLoss()
        softmax_losses = loss_fct(output, labels.view(-1))
        print('softmax losses size:', softmax_losses.shape)

        # bilinear product
        lin_part = torch.matmul(rep_a, self.bilinear_W)
        print('lin_part size:', lin_part.shape)
        bilin_part = torch.multiply(lin_part, rep_b)
        print('bilin_part size:', bilin_part.shape)
        predictions = torch.mean(torch.tanh(torch.sum(bilin_part, dim=-1)), dim=0)
        print('predictions size:', predictions.shape)
        print('predictions:', predictions)
        zero_tensor = torch.tensor([0.0] * predictions.size(0))
        if predictions.is_cuda:
            zero_tensor = zero_tensor.to(predictions.device)
        pure_loss = torch.sum(torch.max(torch.sub(1.0, torch.multiply(predictions, labels)), zero_tensor))

        return contrastive_losses.mean() + pure_loss
