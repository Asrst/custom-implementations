from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
from enum import IntEnum

class CrossEntropyLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss'):
        super().__init__()
        
        self.alpha = alpha
        self.name = name
        self.ignore_index = -1

    def forward(self, inputsuts, targets, attnMasks = None):
        """
        Standard cross entropy loss as defined in pytorch, can be used for 
        single sentence or sentence pair classification tasks.

        To use this loss for training, set ``loss_type`` : **CrossEntropyLoss**
        """
        loss = F.cross_entropy(inputsuts, targets, ignore_index=self.ignore_index) 
        loss *= self.alpha
        return loss


class NERLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss'):
        super().__init__()
        
        self.alpha = alpha
        self.name = name
        self.ignore_index = -1  # return 0 loss for such values

    def forward(self, inputs, target, attnMasks = None):

        """
        This loss is a modified version of cross entropy loss for NER/sequence labelling tasks.
        This loss ignores extra ‘O’ values through attention masks to ignore the loss 
        created for the extra padding of labels till max seq length

        To use this loss for training, set ``loss_type`` : **NERLoss**

        inputs shape would be (batchSize, maxSeqlen, classNum). But for loss calculation
        we need (batchSize, classNum). Hence we will squeeze the batchSize and maxSeqlen together.

        """
        
        if attnMasks is not None:
            nerLoss = attnMasks.view(-1) == 1
            nerlogits = inputs.view(-1, inputs.size(-1))
            nerLabels = torch.where(
                nerLoss, target.view(-1), torch.tensor(self.ignore_index).type_as(target)
            )
            finalLoss = F.cross_entropy(nerlogits, nerLabels, ignore_index=self.ignore_index)

        else:
            finalLoss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), target.view(-1),
                                        ignore_index=self.ignore_index)
 
        finalLoss *= self.alpha
        return finalLoss


class SpanLoss(_Loss):
    def __init__(self, alpha=1.0, name='Span Cross Entropy Loss'):
        super().__init__()

        self.alpha = alpha
        self.name = name
        self.ignore_index = -1

    def forward(self, inputs, target, attnMasks = None):

        """
        This loss is a modified version of cross entropy used to predict the starting
        and the ending index of the span.

        To use this loss for training, set ``loss_type`` : **NERLoss**
        """

        #assert if inputs and target has both start and end values
        assert len(inputs) == 2, "start and end logits should be present for span loss calc"
        assert len(target) == 2, "start and end logits should be present for span loss calc"

        startInputs, endInputs = inputs
        startTarg, endTarg = target
        
        startloss = F.cross_entropy(startInputs, startTarg, ignore_index=self.ignore_index)
        endLoss = F.cross_entropy(endInputs, endTarg, ignore_index=self.ignore_index)

        loss = 0.5 * (startloss + endLoss) * self.alpha
        return loss

