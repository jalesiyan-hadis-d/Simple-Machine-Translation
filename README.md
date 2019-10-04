# Machine_Translation (MT)
thise code implement a simple machine translation. It used different features:
- **GRU**
- **Pre trained Glove Embeding**
- **Encoder-Decoder**
- **Bidirectional**

**Set Hypo parameter:**
- Epoch=25
- Bach size=128
- Learning rate= 0.01
- Validation=0.01
- Dropout=0.5

 Bach size should have smaller size, but it means using GPU. Unfortunately, I could not access to my GPU because of some hardware limitation. So, I built my model with the smallest size that my hardwares limitations allowed.

**The result**
Using LSTM did not show any specific rather than GRU, the reason might be partially independent sentences. So in this case GRU is a better choice because it runs faster with equal accuracy, so in the future models I will use GRU.

Glove can improve our performance very well.

**Next Step:**
currently i am training my model with some pretrained embedding models to check out their results.
