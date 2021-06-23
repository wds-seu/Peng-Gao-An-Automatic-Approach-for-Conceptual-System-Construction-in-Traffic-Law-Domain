from .ContrastiveLoss import *
from .SoftmaxLoss import *


class MultiLabelSoftmaxLoss(nn.Module):
    """
    Combine contrastive loss and softmax loss

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension
    :param num_labels
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 distance_metric = SiameseDistanceMetric.COSINE_DISTANCE,
                 margin: float = 0.5,
                 num_labels: int = 2,
                 seq_len: int = 128):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.model = model
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.distance_metric = distance_metric
        self.margin = margin
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.bilinear_W = nn.Parameter(torch.randn(sentence_embedding_dimension,
                                                   num_labels,
                                                   sentence_embedding_dimension))
        nn.init.orthogonal_(self.bilinear_W)

    def get_bilinear_W(self):
        return self.bilinear_W

    def logsumexp(self, x, dim=None, keepdim=False):
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('-inf')),
            xm,
            xm + torch.log(torch.sum(torch.exp(x-xm), dim, keepdim=True))
        )

        return x if keepdim else x.squeeze(dim)

    def forward(self,
                sentence_features: Iterable[Dict[str, Tensor]],
                labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_a, rep_b = reps

        # contrastive loss
        distances = self.distance_metric(rep_a, rep_b)
        contrastive_losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float()
                                    * F.relu(self.margin - distances).pow(2))

        # bilinear product
        # rep_a: batch_size x sentence_embedding_dimension
        batch_size = rep_a.shape[0]
        # print('rep_a size:', rep_a.shape)
        # print('bilinear_w size:', self.bilinear_W.shape)
        # rep_a size: batch_size x sentence_embedding_dimension
        # bilinear_w size: num_labels x sentence_embedding_dimension
        # lin: batch_size x sentence_embedding_dimension*num_labels
        lin = torch.matmul(rep_a, torch.reshape(self.bilinear_W,
                                                (self.sentence_embedding_dimension, -1)))
        # print('lin size:', lin.shape)
        # lin: batch_size x num_labels x sentence_embedding_dimension
        lin = lin.view(batch_size, self.num_labels, -1)
        # print('lin size after view:', lin.shape)
        # print('rep_b size:', rep_b.shape)
        rep_b = torch.transpose(rep_b, 0, 1)
        # print('rep_b size after transpose:', rep_b.shape)
        # bilin: batch_size x num_labels x batch_size
        bilin = torch.matmul(lin, rep_b)
        # print('bilin size:', bilin.shape)
        # bilin = bilin.view(-1, self.num_labels, self.seq_len)
        # print('bilin size after view:', bilin.shape)
        # pairwise_scores = bilin.view(self.batch_size,
        #                              self.seq_len,
        #                              self.num_labels,
        #                              self.seq_len)
        # print('pairwise_scores size:', pairwise_scores.shape)
        softmax = torch.nn.Softmax(dim=1)
        result = softmax(bilin)
        # result = torch.reshape(result, (0, 1, 3, 2))
        # print('result size:', result.shape)
        result = torch.reshape(result, (batch_size, batch_size, self.num_labels))
        output = self.logsumexp(result, 1)
        # print('output size:', output.shape)

        # softmax loss
        loss_fct = nn.CrossEntropyLoss()
        loss_batch = loss_fct(output, labels.view(-1))
        softmax_losses = torch.mean(loss_batch)

        return contrastive_losses.mean() + softmax_losses if self.num_labels == 2 else softmax_losses
