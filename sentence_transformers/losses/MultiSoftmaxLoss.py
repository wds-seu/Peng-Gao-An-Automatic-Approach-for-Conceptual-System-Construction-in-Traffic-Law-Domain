from .ContrastiveLoss import *
from .SoftmaxLoss import *


class MultiSoftmaxLoss(nn.Module):
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
        super(MultiSoftmaxLoss, self).__init__()
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
        output_features = [self.model(sentence_feature) for sentence_feature in sentence_features]
        reps = [output_feature['sentence_embedding'] for output_feature in output_features]
        toks = [output_feature['token_embeddings'] for output_feature in output_features]
        assert len(reps) == 2
        assert len(toks) == 2
        rep_a, rep_b = reps
        tok_a, tok_b = toks

        # contrastive loss
        distances = self.distance_metric(rep_a, rep_b)
        contrastive_losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float()
                                    * F.relu(self.margin - distances).pow(2))

        # bilinear product
        # tok_a shape: batch_size x seq_len x sentence_embedding_dimension
        batch_size = tok_a.shape[0]
        seq_len = tok_a.shape[1] if tok_a.shape[1] == tok_b.shape[1] \
            else max(tok_a.shape[1], tok_b.shape[1])
        # print('tok_a size:', tok_a.shape)
        # print('bilinear_w size:', self.bilinear_W.shape)
        # toka shape: batch_size*seq_len x sentence_embedding_dimension
        tok_a = tok_a.view(-1, self.sentence_embedding_dimension)
        # print('tok_a view shape:', tok_a.shape)
        # bilinear_w size: sentence_embedding_dimension x num_labels x sentence_embedding_dimension
        # lin: batch_size*seq_len x num_labels*sentence_embedding_dimension
        lin = torch.matmul(tok_a, torch.reshape(self.bilinear_W,
                                                (self.sentence_embedding_dimension, -1)))
        # print('lin size:', lin.shape)
        lin = lin.view(batch_size, seq_len*self.num_labels, -1)
        # print('lin size after view:', lin.shape)
        # print('tok_b size:', tok_b.shape)
        tok_b = torch.transpose(tok_b, 1, 2)
        # print('tok_b size after transpose:', tok_b.shape)
        # bilin shape: batch_size x seq_len*num_labels x seq_len
        bilin = torch.matmul(lin, tok_b)
        # print('bilin size:', bilin.shape)
        # bilin view shape: batch_size*seq_len x num_labels x seq_len
        bilin = bilin.view(-1, self.num_labels, seq_len)
        # print('bilin size after view:', bilin.shape)
        bilin = bilin.view(batch_size, seq_len, self.num_labels, -1)
        # pairwise_scores = bilin.view(self.batch_size,
        #                              self.seq_len,
        #                              self.num_labels,
        #                              self.seq_len)
        # print('pairwise_scores size:', pairwise_scores.shape)
        softmax = torch.nn.Softmax(dim=2)
        result = softmax(bilin)
        # print('result size:', result.shape)
        result = torch.transpose(result, 2, 3)
        output = self.logsumexp(result, 1)
        output = self.logsumexp(output, 1)
        # print('output size:', output.shape)

        # softmax loss
        loss_fct = nn.CrossEntropyLoss()
        loss_batch = loss_fct(output, labels.view(-1))
        softmax_losses = torch.mean(loss_batch)

        return contrastive_losses.mean() + softmax_losses if self.num_labels == 2 else softmax_losses
