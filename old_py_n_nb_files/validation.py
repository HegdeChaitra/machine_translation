from bleu_score import BLEU_SCORE
from define_training_viet import * 


def convert_idx_2_sent(tensor, lang_obj):
    word_list = []
    for i in tensor:
        if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            word_list.append(lang_obj.index2word[i.item()])
    return (' ').join(word_list)

def validation(encoder, decoder, dataloader, loss_fun, lang_en):
    encoder.train(False)
    decoder.train(False)
    pred_corpus = []
    true_corpus = []
    running_loss = 0
    running_total = 0
    bl = BLEU_SCORE()
    for data in dataloader:
        encoder_i = data[0].cuda()
        decoder_i = data[1].cuda()
        bs,sl = encoder_i.size()[:2]
        out, hidden = encode_decode(encoder,decoder,encoder_i,decoder_i)
        loss = loss_fun(out.float(), decoder_i.long())
        running_loss += loss.item()*bs
        running_total += bs
        pred = torch.max(out,dim = 1)[1]
        for t,p in zip(data[1],pred):
            t,p = convert_idx_2_sent(t,lang_en), convert_idx_2_sent(p,lang_en)
            true_corpus.append(t)
            pred_corpus.append(p)
    score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
    return running_loss/running_total, score