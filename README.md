# text-sentiment-transfer-based-on-kewwords

This model implements text sentiment transfer both in sentence level and paragraph level by using keyword generation. Note that Our model does not require any discriminator to provide any constraints to achieve high migration accuracy. 
There are 3 method called: keepNV、keepN、styleword.

In the Yelp resturant review dataset(sentence level) ,all 3 method achieved:

styleword:\
test_fake_acc: 0.7717, test_acc_real: 0.9630, bleu: 18.980, bleu1: 58.800, bleu2: 28.200, bleu3: 13.100, bleu4: 6.300\
keepN：\
test_fake_acc: 0.9361, test_acc_real: 0.9628, \
keepNV:


In the Amazon food review dataset(paragraph level length 30) ,it achieved:\
keepN：\
test_fake_acc: 0.8257, test_acc_real: 0.8992, bleu: 7.480, bleu1: 49.300, bleu2: 14.600, bleu3: 4.600, bleu4: 1.600\
keepNV:\
test_fake_acc: 0.7225, test_acc_real: 0.9000, bleu: 21.070, bleu1: 70.000, bleu2: 34.900, bleu3: 17.800, bleu4: 9.400

In the Amazon food review dataset(paragraph level length 50) ,it achieved:\
keepN：\
test_fake_acc: 0.9180, test_acc_real: 0.9347, bleu: 6.720, bleu1: 47.500, bleu2: 13.000, bleu3: 3.700, bleu4: 1.200\
keepNV:\
test_fake_acc: 0.7675, test_acc_real: 0.9352, bleu: 10.170, bleu1: 34.700, bleu2: 14.200, bleu3: 6.900, bleu4: 3.400




## generated sample

### yelp:
#### negative 2 positive:

| negative review      | positive review     | 
| ---------- | :-----------:  | 
| the people behind the counter were not friendly whatsoever .      | people behind the counter very friendly .     | 
| i would n't come back here though .       | i would come back soon .      | 

           
#### positive 2 negative:

| positive review      | negative review     | 
| ---------- | :-----------:  | 
| great energy and super service .      |terrible energy and poor service .      | 
| it was fabulous !        |i was very disappointed .     | 
                                                                                      

### amazon:
#### negative 2 positive

| negative review      | positive review     | 
| ---------- | :-----------:  | 
| i 've been here three different times and every time regardless of how many customers are in store the wait is ridiculous . they have about 7-9 stations yet only use two or three to help people . waiting 1-2 hours is not ideal for customer service .      | i have been here several times and have been getting a third time . i have been to the first time and the help is always good . they have a nice stations and the people are very nice and friendly . i love the two hours of the customer service . | 

#### positive 2 negative

| positive review      | negative review     | 
| ---------- | :-----------:  | 
| incredible hotel , incredible staff in the perfect location in downtown montreal . the breakfast , whether complementary or through the restaurant , is incredible . the hotel space is trendy , funky and very cool .      |tthis hotel is the worst staff ever . this location is very rude and the breakfast tasted bad . the hotel is very dirty and the space is very small .      | 




## --mdoel training：
run GAN.py
