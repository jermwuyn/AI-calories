import torch


class Metrics:
    def __init__(self, n):
        self.n = n
        self.matrix = torch.zeros((n, n), dtype=torch.int64)
        self.count = self.correct3 = 0
        self.loss = 0.0

    def update(self, logits, labels, loss):
        output, target = logits.detach(), labels.detach()
        pred = output.argmax(1)
        self.matrix += torch.bincount((target * self.n + pred).cpu(), minlength=self.n ** 2).reshape(self.n, self.n)
        self.correct3 += (output.topk(min(3, self.n), dim=1).indices == target[:, None]).any(1).sum().item()
        self.count += len(target)
        self.loss += loss.item() * len(target)

    def result(self, classes):
        m = self.matrix.double()
        tp, support, predicted = m.diag(), m.sum(1), m.sum(0)
        precision = tp / predicted.clamp(min=1)
        recall = tp / support.clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        return {'loss': self.loss / self.count, 'top1_accuracy': (tp.sum() / self.count).item(),
                'top3_accuracy': self.correct3 / self.count, 'macro_f1': f1.mean().item(),
                'per_class': [{'index': i, 'name': c['source_name'], 'support': int(support[i]),
                               'precision': precision[i].item(), 'recall': recall[i].item(), 'f1': f1[i].item()}
                              for i, c in enumerate(classes)], 'confusion_matrix': self.matrix.tolist()}
