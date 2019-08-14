# text-sentiment-transfer-based-on-kewwords

This model implements text sentiment transfer both in sentence level and paragraph level by using keyword generation. 
Note that Our model does not require any discriminator to provide any constraints to achieve high migration accuracy. 
There are 3 method called: keepNV、keepN、styleword.

In the Yelp resturant review dataset(sentence level) ,all 3 method achieved:

styleword:/
test_fake_acc: 0.7717, test_acc_real: 0.9630, bleu: 0.110, bleu1: 10.900, bleu2: 0.300/
keepN：/
test_fake_acc: 0.9361, test_acc_real: 0.9628, bleu: 0.240, bleu1: 11.700, bleu2: 0.400/
keepNV:/
In the Amazon food review dataset(paragraph level length 30) ,it achieved:/

keepN：
test_fake_acc: 0.8257, test_acc_real: 0.8992, bleu: 0.600, b1: 14.300, b2: 1.200, b3: 0.200, b4: 0.000

keepNV:
test_fake_acc: 0.7225, test_acc_real: 0.9000, bleu: 0.230, b1: 13.000, b2: 0.700, b3: 0.100, b4: 0.000

In the Amazon food review dataset(paragraph level length 50) ,it achieved:

keepN：
test_fake_acc: 0.9180, test_acc_real: 0.9347, bleu: 1.340, b1: 17.100, b2: 2.600, b3: 0.500, b4: 0.200

keepNV:
test_fake_acc: 0.7675, test_acc_real: 0.9352, bleu: 0.340, b1: 15.800, b2: 1.200, b3: 0.100, b4: 0.000

--mdoel training：
run GAN.py


generated sample

yelp:

negative 2 positive:
the people behind the counter were not friendly whatsoever .            people behind the counter very friendly . 
i would n't come back here though .                                     i would come back though . 
positive 2 negative:
great energy and super service .                                        terrible energy and poor service . 
it was fabulous !                                                       i was very disappointed . 

amazon:
negative 2 positive
i 've been here three different times and every time regardless of how many customers are in store the wait is ridiculous . they have about 7-9 stations yet only use two or three to help people . waiting 1-2 hours is not ideal for customer service . 
i have been here several times and have been getting a third time . i have been to the first time and the help is always good . they have a nice stations and the people are very nice and friendly . i love the two hours of the customer service .
positive 2 negative
incredible hotel , incredible staff in the perfect location in downtown montreal . the breakfast , whether complementary or through the restaurant , is incredible . the hotel space is trendy , funky and very cool . 
this hotel is the worst staff ever . this location is very rude and the breakfast tasted bad . the hotel is very dirty and the space is very small . 
