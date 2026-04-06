_February 2025_ 

**==> picture [91 x 13] intentionally omitted <==**

# **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features** 

**Michael Tschannen**[*,][†] **, Alexey Gritsenko**[*] **, Xiao Wang**[*] **, Muhammad Ferjad Naeem**[*] **, Ibrahim Alabdulmohsin**[*] **, Nikhil Parthasarathy**[*] **, Talfan Evans**[*,][◦] **, Lucas Beyer**[*,][◦] **, Ye Xia, Basil Mustafa, Olivier Hénaff**[◦] **, Jeremiah Harmsen, Andreas Steiner and Xiaohua Zhai**[*,][◦][,][†] 

Google DeepMind,[*] Core contributor,[†] Project lead,[◦] Work done while at Google DeepMind 

**We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success of the original SigLIP. In this second iteration, we extend the original image-text training objective with several prior, independently developed techniques into a unified recipe—this includes captioningbased pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements on localization and dense prediction tasks. We also train variants which support multiple resolutions and preserve the input’s native aspect ratio. Finally, we train on a more diverse data-mixture that includes de-biasing techniques, leading to much better multilingual understanding and improved fairness. To allow users to trade off inference cost with performance, we release model checkpoints at four sizes: ViT-B (86M), L (303M), So400m (400M), and g (1B).** 

## **1. Introduction** 

Contrastive image-text embedding models trained on billion-scale datasets, as pioneered by CLIP [50] and ALIGN [28], have become the mainstream approach for high-level, semantic understanding of visual data. These models enable fine-grained, zero-shot classification rivaling the quality of supervised methods and enable efficient text-to-image and image-to-text retrieval. Furthermore, they lead to excellent vision-language understanding capabilities when combined with Large Language Models (LLMs) to build Vision-Language Models (VLMs). 

Developing on the success of CLIP, several improvements have been proposed such as recaptioning images [38], adding image-only selfsupervised losses [38, 45], and training with a small decoder for auxiliary tasks such as captioning and localization [32, 62, 67]. At the same time, several groups have released model checkpoints for the open-source community [19, 27, 50, 57, 70]. However, these releases do not include the full breadth of latest improvements into 

a single model, as they all relatively closely follow CLIP’s original approach. Here, building on the SigLIP training recipe [71], we incorporate several improvements from prior work and release a new family of open models[1] that both excel on CLIP’s core capabilities—–zero-shot classification, retrieval, and feature extraction for VLMs—and improve areas where vanilla CLIP-style models lag behind, including localization and extracting dense, semantic representations. 

In summary, SigLIP 2 models provide the following: 

- Strong multilingual vision-language encoders: SigLIP 2 shows excellent performance on English-focused vision-language tasks while providing strong results on multilingual benchmarks with a single model. This enables use in a wide range of languages and cultural contexts. 

- Dense features: We incorporate self- 

1Model checkpoints are available at https://github.com/google-research/big_vision/tree/main/ big_vision/configs/proj/image_text/README_siglip2.md 

_Corresponding author(s): tschannen@google.com_ © 2025 Google DeepMind. All rights reserved 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

   - supervised losses as well as a decoder-based loss, which result in better dense features (e.g. for segmentation and depth estimation) and improve localization tasks (such as referring expression comprehension). 

- Backward compatibility: SigLIP 2 is designed to be backward compatible with SigLIP by relying on the same architecture. This allows existing users to simply swap out the model weights and tokenizer (which is now multilingual) to get improvements on a wide range of tasks. 

- Native aspect ratio and variable resolution: SigLIP 2 also includes a NaFlex variant, which supports multiple resolutions and preserves the native image aspect ratio. These models have the potential to improve aspect sensitive applications such as document understanding. 

- Strong small models: SigLIP 2 further optimizes performance of smaller models (B/16 and B/32 models), by using techniques in distillation via active data curation. 

In the next section we provide a detailed description of the SigLIP 2 training recipe. Sec. 3 presents evaluations of SigLIP 2 models and baselines across a variety of tasks and benchmarks. Finally, Sec. 4 gives a short overview of related work, and conclusions can be found in Sec. 5. 

## **2. Training recipe** 

We combine the original SigLIP training recipe [71] with decoder-based pretraining [60, 62], in addition to self-distillation and masked prediction as in the DINO line of work [9, 47] (see Fig. 1 for an overview). Pretraining an image encoder with a language decoder for captioning and referring expression comprehension was shown to improve OCR capabilities and localization [62], whereas self-distillation and masked prediction leads to better features for dense prediction tasks, zero-shot classification and retrieval [38, 45]. Rather than combining all these techniques in a single run we follow a staged approach as outlined below to manage the computational and memory overhead compared to SigLIP training. 

**==> picture [228 x 192] intentionally omitted <==**

**----- Start of picture text -----**<br>
LocCa loss (100%):<br>AR  - captioning<br>SILC/TIPS loss (20%): Decoder - dense captioning<br>- self-distillation - ref. expressions<br>-<br>masked prediction<br>cross-attn. Sigmoid loss (100%)<br>stop gradient aux. head<br>MAP head<br>EMA<br>Image  Image  Text<br>Encoder Encoder Encoder<br>(teacher)<br>SigLIP (v1)<br>image text<br>**----- End of picture text -----**<br>


Figure 1 | SigLIP 2 adds the captioning-based pretraining from LocCa [62] as well as selfdistillation and masked prediction from SILC [45] and TIPS [38] (during the last 20% of training) to the sigmoid loss from SigLIP [71]. For some variants, the recipe additionally involves fine-tuning with data curation [61] or adaptation to native aspect ratio and variable sequence length [6, 12]. 

In addition to training a set of models and adapting each model separately to different resolutions while distorting the aspect ratio, we also train variants which process images while largely preserving their native aspect ratio like NaViT [12] and support different sequence lengths as FlexiViT [6]. We call this variant NaFlex, described in Sec. 2.4.2. 

Finally, to improve the quality of the smallest models we fine-tune those with implicit distillation via active sample selection, following the approach from [61]. 

## **2.1. Architecture, training data, optimizer** 

For the architecture, we follow SigLIP [71] so that existing users can simply swap out the encoder weights. Specifically, the fixed-resolution variant relies on the standard ViT architecture [15] with learned positional embedding. We use the same architecture for the image and text tower, except for the g-sized vision encoder which is paired with an So400m-sized [1] text encoder. Vision and text representations are pooled using a MAP head (at- 

2 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

tention pooling) [69]. We set the text length to 64 and use the multilingual Gemma tokenizer [22] with vocabulary size 256k, transforming the text to lower case before tokenization. 

We use the WebLI dataset [10] containing 10 billion images and 12 billion alt-texts covering 109 languages. To strike a good balance between quality on English and multilingual visionlanguage benchmarks we compose the mixture such that 90% of the training image-text pairs is sourced from English web pages, and the remaining 10% from non-English web pages, as recommended in [49]. We further apply the filtering techniques from [2] to mitigate data biases in representation and association with respect to sensitive attributes. 

Unless noted otherwise, we use the Adam optimizer with learning rate 10[−][3] , decoupled weight decay 10[−][4] [37], and gradient clipping to norm 1. We set the batch size to 32k and use a cosine schedule with 20k warmup steps, training for a total of 40B examples. Our models are trained on up to 2048 TPUv5e chips [24] using a fullysharded data-parallel strategy (FSDP [72]). 

## **2.2. Training with Sigmoid loss and decoder** 

In the first step of pretraining, we combine SigLIP [71] with LocCa [62] by simply combining the two losses with equal weight. Unlike CLIP [50], which relies on a contrastive loss, SigLIP creates binary classification problems by combining every image embedding with every text embedding in the mini-batch and trains the embeddings to classify matching and nonmatching pairs via logistic regression (sigmoid loss). We use the original implementation and refer to [71] for details. 

For LocCa, we attach a standard transformer decoder with cross-attention to the un-pooled vision encoder representation (before applying the MAP head). The decoder follows the shapes of the text encoder except that we add crossattention layers and reduce the number of layers by a factor of two. Besides image captioning, LocCa also trains for automatic referring expression prediction and grounded captioning. The former amounts to predicting bounding box co- 

ordinates for captions describing specific image regions, whereas the latter involves predicting region-specific captions given bounding box coordinates. Region-caption pairs are automatically annotated by first extracting n-grams from the alttexts and then applying open-vocabulary detection using the recipe from [41]. Additionally, we use the fixed set of object categories from [10] instead of n-grams. For each example, the decoder is trained to predict all three targets (amounting to three decoder forward-passes). The captioning target is predicted with parallel prediction [60] with probability of 50%, i.e. all caption tokens are predicted in parallel from mask tokens, without causal attention mask. Please refer to [62] for more detail. Finally, to reduce memory consumption due to the large vocabulary, we implement a chunked version of the decoder loss. 

For all model sizes, we set the vision encoder patch size to 16 and the image resolution to 256 (resulting in an image representation sequence length of 256). Finally, we note that the decoder only serves for representation learning here and is not part of the model release. 

## **2.3. Training with self-distillation and masked prediction** 

Following SILC [45] and TIPS [38], we augment the training setup described in Sec. 2.2 with local-to-global correspondence learning with self-distillation and masked prediction losses [9, 47, 75] to improve the local semantics of the (un-pooled) feature representation. This representation is typically used for dense prediction tasks like segmentation, depth estimation etc. Concretely, we add two terms to the losses described in Sec. 2.2 as detailed next. 

The first term is the local-to-global consistency loss from [45], in which the vision encoder becomes the student network, which gets a partial (local) view of the training image, and is trained to match the teacher’s representation, derived from the full image. This auxiliary matching task is performed in a high-dimensional feature space computed with a separate MLP head. As is common in the literature, the teacher parameters are obtained as an exponential moving average 

3 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

(EMA) of the student parameters over the previous iterations. We rely on a single global (teacher) view and 8 local (student) views and otherwise follow the augmentations, loss and hyper parameters from [45]. 

The second loss term is the masked prediction objective from [38]. We replace 50% of the embedded image patches in the student network with mask tokens and train the student to match the features of the teacher at masked locations. The loss is then defined identically to the first term (consistency loss), but applied to per-patch features rather than the pooled, image-level representation. Further, both the student and the teacher see the same, global view (up to masking in the student). 

We add these losses at 80% of training completion, initializing the teacher with the student parameters and the remaining additional parameters (heads, mask token and corresponding optimizer parameters) randomly. We use the original image for computing the SigLIP and LocCa losses from the previous section and apply the additional losses on additional augmented views. This is done to ensure that data augmentation does not negatively impact the image-text alignment as recommended by [45]. The weights of the first and the second loss terms are set to 1 and 0.25. Further, to balance model quality on global/semantic and dense tasks, we re-weight the two loss terms by another factor of 0.25, 0.5, 1.0, and 0.5 for the B, L, So400m and g, model sizes, respectively. 

## **2.4. Adaptation to different resolutions** 

## _**2.4.1. Fixed-resolution variant**_ 

To obtain fixed-resolution checkpoints at multiple resolutions, we resume the checkpoints (with sequence length 256 and patch size 16) at 95% of training, resize the positional embedding to the target sequences length (and in some cases the patch embedding from patch size 16 to 14 with the pseudoinverse (PI)-resize strategy from [6]), and resume the training at the target resolution with all losses. We opt for this approach as the common strategy of fine-tuning the final checkpoint with smaller learning rate and with- 

out weight decay [71] did not lead to good results across all sizes and resolutions. 

## _**2.4.2. Variable aspect and resolution (NaFlex)**_ 

NaFlex combines ideas from FlexiViT [6], i.e. supporting multiple, predefined sequence lengths with a single ViT model, and NaViT [12], namely processing images at their native aspect ratio. This enables processing different types of images at appropriate resolution, e.g. using a larger resolution to process document images, while at the same time minimizing the impact of aspect ratio distortion on certain inference tasks, e.g. on OCR. 

Given a patch size and target sequence length, NaFlex preprocesses the data by first resizing the input image such that the height and width after resizing are multiples of the patch size, while 1) keeping the aspect ratio distortion as small as possible and 2) producing a sequence length of at most the desired target sequence length. The resulting distortion in width and height is at most `(patch_size-1)/width` and `(patch_size-1)/height` , respectively, which tends to be small for common resolutions and aspect ratios. Note that NaViT incurs the same type of distortion. After resizing, the image is split into a sequence of patches, and patch coordinates as well as a mask with padding information is added (to handle the case where the actual sequence length is smaller than the target length). 

To process different sequence lengths (and aspect ratios) with a ViT, we bilinearly resize (with anti-aliasing) the learned positional embedding to the target, non-square patch grid for the resized input image. We set the length of the learned positional embedding to 256, assuming a 16 × 16 patch grid before resizing. When the sequence length after resizing is smaller than the target sequence length, the attention layers (including the MAP head) are masked to ignore the extra padding tokens. 

As for the fixed-resolution, adapted variants, we start from the default checkpoints trained with the setup described in Sec. 2.2, i.e. with nonaspect preserving resize to 256px, resulting in a sequence length of 256. We take the checkpoint at 90% training completion, then switch to aspect- 

4 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

|ViT|Res.<br>Seq.<br>Model||ImageNet-1k|ImageNet-1k|ImageNet-1k|10s.|COCO<br>T→I<br>I→T|COCO<br>T→I<br>I→T|Flickr<br>T→I<br>I→T|Flickr<br>T→I<br>I→T|XM3600<br>T→I<br>I→T|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||val|v2|ReaL<br>ObjNet|||T→I||T→I|||
|B/32|224<br>49<br>MetaCLIP [66]|67.7|59.6|–|52.8|–|46.6|–|72.9|–|–<br>–|
||256<br>64<br>OpenCLIP [27]<br>SigLIP 2|72.8|64.8|–|59.6|–|39.9|57.9|64.9|84.8|–<br>–|
|||**74.0**|**66.9**|**81.4**|**66.1**|**66.6**|**47.2**|**63.7**|**75.5**|**89.3**|**38.3**<br>**49.0**|
|B/16|224<br>196<br>CLIP [50]<br>OpenCLIP [27]<br>MetaCLIP [66]<br>EVA-CLIP [57]<br>SigLIP [71]<br>DFN [19]<br>SigLIP 2|68.3<br>70.2<br>72.4<br>74.7<br>76.2<br>76.2|61.9<br>62.3<br>65.1<br>67.0<br>69.5<br>68.2|–<br>–<br>–<br>–<br>82.8<br>–|55.3<br>56.0<br>60.0<br>62.3<br>70.7<br>63.2|–<br>–<br>–<br>–<br>69.9<br>–|33.1<br>42.3<br>48.9<br>42.2<br>47.2<br>51.9|52.4<br>59.4<br>–<br>58.7<br>64.5<br>–|62.1<br>69.8<br>77.1<br>71.2<br>77.9<br>77.3|81.9<br>86.3<br>–<br>85.7<br>89.6<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>22.4<br>29.3<br>–<br>–|
|||78.2|71.4|84.8|73.6|72.1|52.1|68.9|80.7|93.0|40.3<br>50.7|
||256<br>256<br>SigLIP [71]<br>SigLIP 2|76.7|70.1|83.1|71.3|70.3|47.4|65.1|78.3|91.1|22.5<br>29.9|
|||79.1|72.5|85.4|74.5|73.1|53.2|69.7|81.7|94.4|40.7<br>51.0|
||384<br>576<br>SigLIP [71]<br>SigLIP 2|78.6|72.0|84.6|73.8|72.7|49.7|67.5|80.7|92.2|23.3<br>30.3|
|||80.6|73.8|86.2|77.1|74.7|54.6|**71.4**|83.8|94.9|41.2<br>51.6|
||512<br>1024<br>SigLIP [71]<br>SigLIP 2|79.2|72.9|84.9|74.8|73.3|50.4|67.6|81.6|92.5|23.5<br>30.5|
|||**81.2**|**74.5**|**86.7**|**77.8**|**75.2**|**55.2**|71.2|**84.5**|**95.5**|**41.4**<br>**52.0**|
|L/14|224<br>256<br>OpenCLIP [27]<br>CLIP [50]<br>MetaCLIP [66]<br>CLIPA-v2 [33]<br>EVA-CLIP [57]<br>DFN [19]|74.0<br>75.5<br>79.2<br>79.7<br>79.8<br>**82.2**|61.1<br>69.0<br>72.6<br>72.8<br>72.9<br>**75.7**|–<br>–<br>–<br>–<br>–<br>–|66.4<br>69.9<br>74.6<br>71.1<br>**75.3**<br>74.8|–<br>–<br>–<br>–<br>–<br>–|46.1<br>36.5<br>55.7<br>46.3<br>47.5<br>**59.6**|62.1<br>56.3<br>–<br>**64.1**<br>63.7<br>–|75.0<br>65.2<br>83.3<br>73.0<br>77.3<br>**84.7**|88.7<br>85.2<br>–<br>89.1<br>**89.7**<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|
|L/16|256<br>256<br>SigLIP [71]<br>SigLIP 2|80.5|74.2|85.9|77.9|76.8|51.2|69.6|81.3|92.0|30.9<br>40.1|
|||82.5|76.8|87.3|83.0|78.8|54.7|71.5|84.1|94.5|46.5<br>56.5|
||384<br>576<br>SigLIP [71]<br>SigLIP 2|82.1|75.9|87.1|80.9|78.7|52.8|70.5|82.6|92.9|31.4<br>39.7|
|||83.1|77.4|87.6|84.4|79.5|**55.3**|71.4|85.0|95.2|47.1<br>56.3|
||512<br>1024<br>SigLIP 2|**83.5**|**77.8**|**87.7**|**84.6**|**79.6**|55.2|**72.1**|**85.3**|**95.8**|**47.4**<br>**56.7**|
|So/14|224<br>256<br>SigLIP [71]<br>SigLIP 2|82.2|76.0|87.1|80.5|78.2|50.8|69.0|76.6|90.7|16.0<br>22.8|
|||83.2|77.7|87.8|84.6|79.5|55.1|71.5|84.3|94.6|47.9<br>57.5|
||384<br>729<br>SigLIP [71]<br>SigLIP 2|83.2|77.1|87.5|82.9|79.4|52.0|70.2|80.5|93.5|17.8<br>26.6|
|||**84.1**|**78.7**|**88.1**|**86.0**|**80.4**|**55.8**|**71.7**|**85.7**|**94.9**|**48.4**<br>**57.5**|
|So/16|256<br>256<br>mSigLIP [71]<br>SigLIP 2|80.8|74.1|86.1|79.5|77.1|49.4|68.6|80.0|92.1|**50.0**<br>**62.8**|
|||83.4|77.8|87.7|84.8|79.7|55.4|**71.5**|84.4|94.2|48.1<br>57.5|
||384<br>576<br>SigLIP 2|84.1|78.4|**88.1**|85.8|80.4|**56.0**|71.2|85.3|**95.9**|48.3<br>57.5|
||512<br>1024<br>SigLIP 2|**84.3**|**79.1**|88.1|**86.2**|**80.5**|56.0|71.3|**85.5**|95.4|48.3<br>57.6|
|H/14|224<br>256<br>MetaCLIP [66]<br>DFN [19]|80.5<br>**83.4**|74.1<br>**77.3**|–<br>–|76.5<br>**76.5**|–<br>–|57.5<br>**63.1**|–<br>–|85.0<br>**86.5**|–<br>–|–<br>–<br>–<br>–|
|g/16|256<br>256<br>SigLIP 2|84.5|79.2|88.3|87.1|82.1|55.7|72.5|85.3|95.3|48.2<br>**58.2**|
||384<br>576<br>SigLIP 2|**85.0**|**79.8**|**88.5**|**88.0**|**82.5**|**56.1**|**72.8**|**86.0**|**95.4**|**48.6**<br>57.9|



Table 1 | Zero-shot classification, 10-shot (10s) classification (on the validation set), and retrieval performance (recall@1) of SigLIP 2 along with several baselines. SigLIP 2 outperforms the baselines— often by a large margin—despite being multilingual. Note that DFN [19] relies on a data filtering network fine-tuned on ImageNet, COCO, and Flickr. 

preserving resizing and uniformly sampling a sequence length from {128 _,_ 256 _,_ 576 _,_ 784 _,_ 1024} per mini-batch. At the same time we stretch the learning rate schedule corresponding to the last 10% by a factor 3 _._ 75 to ensure that each resolution is trained for sufficiently many examples. For the largest sequence length we further half the batch size and double the number of training steps to avoid out-of-memory errors. 

To keep implementation and computation complexity manageable, we do not apply selfdistillation and masked prediction from Sec. 2.3. 

## **2.5. Distillation via active data curation** 

To maximize performance of the smallest fixedresolution models (ViT-B/16 and ViT-B/32), we distill knowledge from a teacher (reference) 

5 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**==> picture [471 x 191] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.8 Text → Image Retrieval<br>SigLIP<br>0.6 SigLIP 2<br>mSigLIP<br>0.4<br>0.2<br>0.0<br>Image → Text Retrieval<br>0.8 SigLIP<br>SigLIP 2<br>0.6<br>mSigLIP<br>0.4<br>0.2<br>0.0<br>MaoriCusco QuechuaTelugu SwahiliBengali HindiFilipinoGreek, ModernThai FinnishJapaneseChineseHebrewCzechCroatianArabicUkrainianHungarianEnglishRomanian, MoldovanTurkish PersianKoreanSwedishNorwegianDutch, FlemishDanish Spanish, CastilianPolish PortugueseRussianVietnameseIndonesianItalianFrenchGerman avg<br>recall@1<br>recall@1<br>**----- End of picture text -----**<br>


Figure 2 | Per-language image-text retrieval performance for SigLIP, SigLIP 2 and mSigLIP on Crossmodal-3600 [58]. SigLIP 2 almost matches the performance of mSigLIP (SigLIP trained on multilingual data) despite performing substantially better on English vision-language tasks (Table 1). 

model during a short fine-tuning stage. We lower the learning rate to 10[−][5] , remove weight-decay, and continue training these models for an additional 4B examples using just the sigmoid imagetext loss. During this stage, we perform implicit “distillation through data” using the ACID method proposed in [61]. Briefly, at every training step, the teacher model and the current learner model are used to score examples by their “learnability” [42]. These scores are then used to jointly select an optimal batch of size 32k from a larger super-batch [16]. Here, we select data with a filtering ratio of 0.5 (i.e. super-batch size of 64k) to balance gains from curation with training compute. For the B/32 model, we find leveraging a filtering ratio of 0.75 is worth the extra cost. 

We note that the authors in [61] suggest that the best performance is achieved with ACED, a method that combines ACID with explicit softmaxdistillation (using a second teacher trained on more diverse data). However, here we propose a way to adapt ACID to capture these benefits _without the need for explicit distillation_ , saving significant amounts of compute. Specifically, instead of utilizing two separate teacher models, we take a single strong teacher trained on the diverse data (in this case, the SigLIP 2 So400m model) and fine-tune it for 1B examples on the high-quality curated dataset from [16]. We then 

use this fine-tuned teacher model in the ACID method, as described above. Because this teacher blends diverse knowledge of concepts from pretraining, with knowledge of what is high-quality (from the curated dataset), the implicit distillation of ACID alone is sufficient to recover the benefits of ACED. 

## **3. Experiments and results** 

## **3.1. Zero-shot classification and retrieval** 

In Table 1 we report the performance of SigLIP 2 along with baselines on common zero-shot classification (ImageNet [13] ObjectNet [4], ImageNetv2 [53], ImageNet ReaL [5]) and image-text retrieval benchmarks. SigLIP 2 performs better than SigLIP and other (open-weight) baselines across the board, despite supporting many languages unlike the baselines (except mSigLIP [71]). Note that DFN [19], which comes closest to SigLIP 2 on these benchmarks, uses a network fine-tuned on ImageNet, COCO, and Flickr (i.e. the main benchmarks in Table 1) as a filter to improve data quality. SigLIP 2’s improvements over the baselines are particularly significant for the B-sized models owing to distillation (Sec. 2.5). Moreover, we observe the common scaling trends as a function of image resolution and model size. 

6 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**==> picture [471 x 285] intentionally omitted <==**

**----- Start of picture text -----**<br>
INet 0-shot INet v2 0-shot INet ReaL 0-shot ObjectNet 0-shot COCO R@1 T→I<br>85<br>55<br>75 80<br>80 85<br>50<br>70<br>75 70<br>80<br>65 45<br>COCO R@1 I→T TextCaps R@1 T→I TextCaps R@1 I→T HierText R@1 T→I HierText R@1 I→T<br>70 45<br>10.0<br>70 15<br>40<br>7.5<br>65<br>35 10<br>5.0<br>60<br>60 30<br>64 256 576 784 1024<br>sequence length<br>SciCap R@1 T→I SciCap R@1 I→T Screen2Words R@1 T→I Screen2Words R@1 I→T<br>30<br>30 Model<br>30<br>15 SigLIP 2 (NaFlex)<br>20<br>20 20 SigLIP 2<br>10 ViT<br>10 10 B/16<br>So400m/16<br>64 256 576 784 1024 64 256 576 784 1024 64 256 576 784 1024 64 256 576 784 1024<br>sequence length sequence length sequence length sequence length<br>**----- End of picture text -----**<br>


Figure 3 | Comparing the NaFlex (a single checkpoint per model size supporting native aspect ratio and variable sequence length/resolution) and the standard square-input SigLIP 2 variants which use a separate checkpoint for each sequence length/resolution. The sequence lengths annotated on the x-axis correspond to training sequence lengths for NaFlex. NaFlex interpolates fairly well between training resolutions, but does not extrapolate well (not shown). 

Table 1 and Figure 2 further show the multilingual retrieval performance on Crossmodal-3600 (XM3600) [58] covering 36 languages. SigLIP 2’s recall exceeds that of SigLIP by a large margin, while only lagging slightly behind mSigLIP, which in turn performs substantially worse than SigLIP and SigLIP 2 on English-focused benchmarks. 

in particular for small sequence lengths (and hence resolutions) which tend to suffer more from aspect ratio distortion. On benchmarks predominantly based on natural images, the standard B-sized variant outperforms NaFlex, arguably thanks to the distillation step, whereas for the So400m architecture the two are on par. This is remarkable since the standard variant also benefits from the self-distillation stage (Sec. 2.3). 

## _**3.1.1. NaFlex variant**_ 

Fig. 3 compares the fixed-resolution squareaspect ratio (standard) SigLIP 2 with the aspectpreserving NaFlex variant (one checkpoint for all sequence lengths) as a function of the sequence length. In addition to the retrieval benchmarks listed in the previous section, we add a range of OCR/document/screen-focused imagetext benchmarks, namely TextCaps [55], HierText [36], SciCap [26] and Screen2Words [63]. The NaFlex variant outperforms the standard variant on the majority of these retrieval benchmarks, 

## **3.2. SigLIP 2 as a vision encoder for VLMs** 

A popular use case for vision encoders like CLIP and SigLIP is to extract visual representations for VLMs [3, 7, 32, 35, 39, 48, 59]. The common paradigm combines a pretrained vision encoder with a pretrained LLM and does multimodal training on a rich mixture of vision language tasks. To evaluate the performance of SigLIP 2 in this application, we develop a recipe similar to that of PaliGemma 2 [56]. Concretely, we combine 

7 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**==> picture [471 x 500] intentionally omitted <==**

**----- Start of picture text -----**<br>
AI2D AOKVQA-DA (val) AOKVQA-MC (val) COCO-35L (avg34) COCO-35L (en) COCOcap<br>84 142<br>144<br>78<br>64 82 1 14 140<br>142<br>76<br>62 80 1 12<br>138 1 40<br>74<br>78 1 10<br>60 1 36 138<br>CountBenchQA DocVQA (val) GQA InfoVQA (val) NLVR2 NoCaps<br>85 70 68 122<br>60 35 92<br>67<br>80 50 120<br>30 91<br>66<br>40<br>118<br>75 30 65 25 90<br>OCR-VQA OKVQA RefCOCO (testA) RefCOCO (testB) RefCOCO (val) RefCOCO+ (testA)<br>76<br>75<br>75 75<br>74 64<br>75<br>70<br>72 63 70<br>70<br>70 62 70 65<br>RefCOCO+ (testB) RefCOCO+ (val) RefCOCOg (test) RefCOCOg (val) ST-VQA (val) SciCap<br>80<br>180<br>65 70 70 70 70<br>170<br>65 65 65 60 160<br>60<br>50<br>ScienceQA Screen2Words TallyQA (complex) TallyQA (simple) TextCaps TextVQA (val)<br>97 86 150<br>115 72 70<br>84 1 40<br>96 70<br>110 82 1 30 60<br>95 68 80 120<br>50<br>1 10<br>VQAv2 (minival) VizWizVQA (val) WidgetCap XM3600 (avg35) XM3600 (en) Average<br>78 90<br>150 41 81<br>77<br>84 80<br>76 40 85<br>140 79<br>82 75 78<br>39 80<br>74 1 30<br>SigLIP L/16 256px SigLIP2 L/16 256px SigLIP2 So400m/14 224px SigLIP2 So400m/14 384px<br>AIMv2 L/14 224px SigLIP So400m/14 224px SigLIP So400m/14 384px<br>**----- End of picture text -----**<br>


Figure 4 | Comparison of different vision encoders after training a Gemma 2 LLM for 50M steps with a frozen vision encoder (PaliGemma [7] stage 1), followed by fine-tuning the VLM on individual datasets (PaliGemma stage 3). SigLIP 2 performs better than SigLIP and AIMv2 [20] for different model sizes and resolutions. Same data as in Table 6. 

SigLIP 2 vision encoders and baselines with the Gemma 2 2B LLM [23] and train the LLM on 50M examples of the Stage 1 training mix from [7, 56] involving captioning, OCR, grounded captioning, visual question answering, detection, and instance segmentation (the annotations for the last 

4 tasks are machine-generated, see [7, Sec. 3.2.5] for details). We keep the vision encoder frozen (which has essentially no impact on quality [7, Sec. 5.4]) and reduce training duration to reflect a typical open model use case. The resulting VLM is then fine-tuned on a broad range of down- 

8 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

|Model<br>ViT<br>Res.|Segmentation↑<br>PASCAL<br>ADE20k|Depth↓<br>NYUv2<br>NAVI|Normals↓<br>NYUv2<br>NAVI|
|---|---|---|---|
|CLIP [50]<br>L/14<br>224<br>OpenCLIP [27]<br>G/14<br>224<br>SigLIP [71]<br>So/14<br>224<br>SigLIP 2<br>So/14<br>224|74.5<br>39.0<br>71.4<br>39.3<br>72.0<br>37.6<br>**77.1**<br>**41.8**|0.553<br>0.073<br>0.541<br>–<br>0.576<br>0.083<br>**0.493**<br>**0.067**|**24.3**<br>25.5<br>–<br>–<br>25.9<br>26.0<br>24.9<br>**25.4**|
|SigLIP [71]<br>So/14<br>384<br>SigLIP 2<br>So/14<br>384|73.8<br>40.8<br>**78.1**<br>**45.4**|0.563<br>0.069<br>**0.466**<br>**0.064**|24.1<br>25.4<br>**23.0**<br>**25.0**|



Table 2 | Probing the frozen SigLIP 2 representation for a range of dense prediction tasks (metrics: segmentation: mIoU; depth: RMSE; normals; angular RMSE). SigLIP 2 outperforms several other popular open-weight models, often by a significant margin. 

stream tasks with the transfer settings from [56]. To understand the effect of the input resolution we perform experiments at resolution 224 or 256 (for models with patch size 14 and 16, respectively, to extract 256 image tokens) and 384px, but unlike [7, 56] we repeat stage 1 at 384px rather than starting from the 224px variant. 

Fig. 4 shows the results after fine-tuning for each dataset. Overall, SigLIP 2 clearly outperforms SigLIP across resolutions and model size. For an L-sized vision encoder, SigLIP 2 also outperforms the recently released AIMv2 model [20]. The data from Fig. 4 can also be found in Table 6. 

## **3.3. Dense prediction tasks** 

## _**3.3.1. Semantic segmentation, depth estimation, surface normal estimation**_ 

We adopt the evaluation protocol from [38] and probe the frozen SigLIP 2 representation, either with a linear layer or with a DPT decoder [52], on six benchmarks spanning semantic segmentation, monocular depth estimation, and surface normal estimation (see [38, Sec. 4.1] for details on the protocol and hyper parameters). Note, we make one (necessary) change: where the original method concatenates the CLS token to each of the patch feature vectors, we concatenate the output embedding of the MAP head instead, as we use a MAP head instead of a CLS token. The results in Table 2 indicate that SigLIP 2 outperforms several previous open, CLIP-style vision encoders, including SigLIP, often by a significant margin. 

## _**3.3.2. Open-vocabulary segmentation**_ 

Open-vocabulary segmentation aims to develop models that can segment any novel classes beyond a fixed training vocabulary. Here, we evaluate SigLIP 2’s performance on this task. We use Cat-Seg [11] as a framework and compare performance across different models as proposed in [45]. We train Cat-Seg on COCOStuff-164k [8] with 172 classes and then test it on various representative datasets with different vocabularies: ADE20k [73, 74] with 847 or 150 classes (A-847/A-150), Pascal Context (PC459/PC-59) [43], and Pascal VOC (VOC-20/VOC21) [17]. The results can be found in Table 3. We observe that the SigLIP 2 at L/16 improves on SigLIP and even surpasses the much bigger OpenCLIP G/14 model [27]. 

## **3.4. Localization tasks** 

## _**3.4.1. Referring expression comprehension**_ 

To probe the referring expression comprehension capabilities of SigLIP 2 on different RefCOCO variants [29, 68] we apply the evaluation protocol from [62]. We attach a 6-layer transformer decoder to the un-pooled, frozen vision encoder representation via cross-attention and train it from scratch on a mix of all RefCOCO variants (see [62] for details). The results in Table 5 show that SigLIP 2 outperforms SigLIP as well as CLIP and pretraining via image captioning (Cap) by a large margin, across resolutions and model sizes. This can be attributed to the decoder-based pretraining, as described in Sec. 2.2. SigLIP 2 is only outperformed LocCa, which we hypothesize 

9 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

|Model|ViT|A-847|PC-459|A-150|PC-59|VOC-20|VOC-21|
|---|---|---|---|---|---|---|---|
|CLIP [50]|L/16|10.8|20.4|31.5|62.0|96.6|81.8|
|OpenCLIP [27]|G/14|13.3|21.4|36.2|61.5|**97.1**|81.4|
|SigLIP [71]|L/16|14.0|23.9|37.5|61.6|96.1|81.1|
|SigLIP 2|L/16|**14.3**|**24.1**|**38.8**|**62.4**|97.0|**82.3**|



Table 3 | We use Cat-Seg [11] to compare open-vocabulary segmentation performance (mIoU) of several models similar to [45]. We observe that SigLIP 2 offers respectable improvements over comparable and even bigger models. 

might be due to the fact that SigLIP 2 is pretrained on multilingual data. LocCa, on the other hand, is trained on text only from English web sites. Finally, note that we expect significant improvements when using the decoder from pretraining as observed for LocCa. 

||ViT<br>B/16<br>So/14|Model<br>SigLIP<br>SigLIP<br>SigLIP<br>SigLIP|2<br>2|COCO (AP)<br>42.2<br>**42.8**<br>44.3<br>**45.2**|LVIS (AP)<br>33.0<br>**34.4**<br>39.5<br>**40.5**|LVIS (APr)<br>31.0<br>**32.7**<br>40.9<br>**42.3**|
|---|---|---|---|---|---|---|



Table 4 | Fine-tuned SigLIP and SigLIP 2 for openvocabulary detection via OWL-ViT [40]. 

## _**3.4.2. Open-vocabulary detection**_ 

OWL-ViT [40] is a popular method to adapt CLIPstyle vision-language models to open-vocabulary detection. Here, we apply this approach to SigLIP and SigLIP 2 models, closely following the data and optimizer configuration from [40]. The results in Table 4 show that SigLIP 2 achieves better performance than SigLIP on the two popular benchmarks COCO [34] and LVIS [25]. The relative improvement is most pronounced for the LVIS rare categories. Further, the results here are better than those in [40] which is likely because [40] used CLIP rather than SigLIP. 

## **3.5. Cultural diversity and fairness** 

Besides the improvement in model quality in SigLIP 2 compared to its predecessor, SigLIP 2 is also more inclusive in two aspects. First, we follow the recommendations of [49] and utilize a training mixture comprising both English and multilingual data to enhance cultural diversity. Second, to address potential societal biases in the training data, we integrate the data de-biasing techniques from [2]. These techniques are applied to mitigate biases in both first-order statistics, such as disparities in gender representation, and second-order statistics, such as biased associations between gender and occupation. Next, we present the evaluation results. 

**Cultural Diversity** To evaluate for cultural diversity, we report the zero-shot classification accuracy results using Dollar Street [54], GeoDE [51], and Google Landmarks Dataset v2 (GLDv2) [65]. We also include 10-shot geolocalization using Dollar Street and GeoDE, as proposed in [49]. For zero-shot evaluation on Dollar Street, we implement the methodology outlined in [54], mapping 96 topics within the dataset to corresponding ImageNet classes. This process results in a subset of 21K images for our analysis. 

Fig. 5 shows a set of representative results (full results are shown in Appendix C). We observe an improvement in these metrics in SigLIP 2 compared to SigLIP for the same model size and resolution, and the improvements are particularly significant in geolocalization tasks. For instance, 10-shot geolocalization accuracy in GeoDE (region) improves from 36.2% for SigLIP L/16 at 256px to 44.4% in SigLIP 2. Similarly, 0-shot accuracy on Dollar Street improves from 52.1% to 55.2% in the same models. 

**Fairness** In terms of fairness, we report two metrics. The first is “representation bias,” as defined in [2], which measures the tendency in the model to associate a random object (such as cars) with a particular gender group. As shown in Fig. 6, SigLIP 2 is _significantly_ better than SigLIP. 

10 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

|ViT|Seq.<br>Model|RefCOCO|testB|RefCOCO+<br>val<br>testA<br>testB|RefCOCO+<br>val<br>testA<br>testB|RefCOCOg<br>val-u<br>test-u|
|---|---|---|---|---|---|---|
|||val<br>testA||val|||
|B|256<br>SigLIP [71]<br>SigLIP 2|64.05<br>70.10|57.89|55.77|63.57<br>47.51|59.06<br>60.33|
|||83.76<br>86.21|79.57|74.26|79.85<br>65.83|77.25<br>77.83|
||576<br>SigLIP [71]<br>SigLIP 2|67.17<br>72.94|60.94|59.09|67.26<br>50.22|61.98<br>62.64|
|||**85.18**<br>**87.92**|**80.53**|**76.08**|**82.17**<br>**67.10**|**79.08**<br>**79.60**|
|L|256<br>Cap [60]<br>CapPa [60]<br>CLIP [50]<br>SigLIP [71]<br>SigLIP 2<br>LocCa [62]|60.64<br>65.47<br>64.17<br>69.90<br>65.21<br>71.28<br>67.33<br>72.40|56.17<br>58.25<br>58.17<br>61.21|52.56<br>56.14<br>57.53<br>59.57|58.32<br>45.99<br>63.68<br>48.18<br>66.44<br>47.77<br>67.09<br>51.08|56.75<br>57.99<br>58.90<br>59.91<br>59.32<br>60.24<br>61.89<br>62.90|
|||86.04<br>89.02|81.85|77.29|83.28<br>70.16|80.11<br>80.78|
|||**88.34**<br>**91.20**|**85.10**|**79.39**|**85.13**<br>**72.61**|81.69<br>**82.64**|
||576<br>SigLIP [71]<br>SigLIP 2|70.76<br>76.32|63.79|63.38|71.48<br>54.65|64.73<br>65.74|
|||87.28<br>90.29|82.85|79.00|85.00<br>70.92|**81.84**<br>82.15|
|So|256<br>SigLIP [71]<br>SigLIP 2|64.68<br>71.23|58.40|57.43|66.06<br>49.38|59.66<br>60.88|
|||86.42<br>89.41|82.48|77.81|84.36<br>70.67|80.83<br>81.27|
||729<br>SigLIP [71]<br>SigLIP 2|67.66<br>74.12|62.36|60.74|69.73<br>52.12|62.61<br>63.24|
|||**87.88**<br>**91.13**|**83.59**|**80.06**|**86.30**<br>**72.66**|**82.68**<br>**83.63**|
|g|256<br>SigLIP 2|87.31<br>90.24|83.25|79.25|85.23<br>71.60|81.48<br>82.14|
||576<br>SigLIP 2|**88.45**<br>**91.53**|**84.95**|**80.44**|**87.09**<br>**73.53**|**83.12**<br>**84.14**|



Table 5 | Comparing SigLIP 2 models with SigLIP and other baselines from the literature on referring expression comprehension (Acc@0.5). For matching model size and sequence length (seq.) SigLIP 2 models outperform SigLIP models substantially. SigLIP 2 is only outperformed by LocCa, which uses the same decoder-based loss, but is trained on captions from English language websites only. 

For instance, while SigLIP L/16 at 256px has a representation bias of about 35.5%—meaning it prefers to associate random images with “men” over “women” more than 85.5% of the time— SigLIP 2 of the same size and resolution has a representation bias of 7.3% only. In addition, larger models tend to exhibit less representation bias than smaller models, in agreement with the earlier findings in [2]. 

We also investigate the Dollar Street 0-shot results by income level and the GeoDE results by geographic region as [49]. However, in this context we only observe very minor benefits, or no benefits when comparing SigLIP and SigLIP 2 models of matching size and resolution (some results shown in Table 9). 

## **4. Related work** 

Contrastive pretraining as popularized by CLIP [50] and ALIGN [28] has become the dominant approach for learning high-level, semantic, visual representations that perform well on classification and retrieval, as vision encoders for VLMs [3, 7, 32, 35, 39, 48, 59] 

and open-vocabulary tasks including detection [30, 40, 41] and segmentation [11, 14]. Besides the original CLIP release, several projects have released open-weight contrastive models [19, 27, 33, 57, 66, 71]. At a high level, these works follow training methods that are relatively close to the original CLIP method, mainly [71] proposing modified loss functions and [19, 66] targeting data quality and filtering. 

More generally, a large number of modifications and improvements to contrastive training have been proposed in the literature. [16, 19, 21, 61, 66] study filtering techniques to improve data quality. With a similar motivation, [18, 31, 38, 46] re-caption training images with VLMs to improve the caption quality and hence the quality of the training signal. Another promising area has been to modify or augment the loss function. [38, 44, 45] combine CLIP with selfsupervised losses. Another popular approach is to add a language decoder to train with captioning as an auxiliary task [32, 67]. Captioning as a standalone representation learning task has attracted less attention, but can produce visual representations competitive with contrastive training [20, 60, 62, 64]. 

11 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**==> picture [229 x 127] intentionally omitted <==**

**----- Start of picture text -----**<br>
10-shot 0-shot<br>40<br>80<br>20 60<br>40<br>0<br>SigLIP B/16 SigLIP 2 B/16 SigLIP L/16 SigLIP 2 L/16<br>Dollar StreetGeoDE (country)GeoDE (region) Dollar Street GLDv2 GeoDE<br>**----- End of picture text -----**<br>


Figure 5 | 10-shot and 0-shot accuracy for geographically diverse object classification tasks (Dollar Street, GeoDE), as well as geolocalization (GeoDE country/region) and landmark localization (GLDv2) tasks. SigLIP 2 consistently performs better than SigLIP (see Table 8 for additional results). 

## **5. Conclusion** 

In this work, we introduced SigLIP 2, a family of open-weight multilingual vision-language encoders that builds on the success of SigLIP. By incorporating a combination of techniques such as decoder-based pretraining, self-supervised losses, and active data curation, SigLIP 2 achieves significant improvements in zero-shot classification, transfer performance as a vision encoder in VLMs, and in localization and dense prediction tasks. Furthermore, thanks to training on multilingual data and applying de-biasing filters, SigLIP 2 attains more balanced quality across culturally diverse data. Finally, the NaFlex variant enables the model to support multiple resolutions with a single model checkpoint, while preserving the native image aspect ratio. We hope that our SigLIP 2 release will enable many exciting applications within the open-source community. 

**Acknowledgments** We would like to thank Josip Djolonga, Neil Houlsby, Andre Araujo, Kevis Maninis, and Phoebe Kirk for discussions and feedback on this project. We also thank Joan Puigcerver, André Susano Pinto, and Alex Bewley for infrastructure contributions to the `big_vision` code base, which were helpful for this project. 

**==> picture [184 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
SigLIP B/16<br>SigLIP 2 B/16<br>SigLIP L/16<br>SigLIP 2 L/16<br>0 10 20 30<br>Representation bias<br>**----- End of picture text -----**<br>


Figure 6 | Representation bias (association of random objects with gender; lower is better) for different models. 

## **References** 

- [1] I. Alabdulmohsin, X. Zhai, A. Kolesnikov, and L. Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. In _NeurIPS_ , 2023. 

- [2] I. Alabdulmohsin, X. Wang, A. P. Steiner, P. Goyal, A. D’Amour, and X. Zhai. Clip the bias: How useful is balancing data in multimodal learning? In _ICLR_ , 2024. 

- [3] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. QwenVL: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv:2308.12966_ , 2023. 

- [4] A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and B. Katz. Objectnet: A large-scale biascontrolled dataset for pushing the limits of object recognition models. _NeurIPS_ , 2019. 

- [5] L. Beyer, O. J. Hénaff, A. Kolesnikov, X. Zhai, and A. v. d. Oord. Are we done with imagenet? _arXiv:2006.07159_ , 2020. 

- [6] L. Beyer, P. Izmailov, A. Kolesnikov, M. Caron, S. Kornblith, X. Zhai, M. Minderer, M. Tschannen, I. Alabdulmohsin, and F. Pavetic. Flexivit: One model for all patch sizes. In _CVPR_ , 2023. 

- [7] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, T. Unterthiner, D. Keysers, 

12 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

   - S. Koppula, F. Liu, A. Grycner, A. Gritsenko, N. Houlsby, M. Kumar, K. Rong, J. Eisenschlos, R. Kabra, M. Bauer, M. Bošnjak, X. Chen, M. Minderer, P. Voigtlaender, I. Bica, I. Balazevic, J. Puigcerver, P. Papalampidi, O. Henaff, X. Xiong, R. Soricut, J. Harmsen, and X. Zhai. PaliGemma: A versatile 3B VLM for transfer. _arXiv:2407.07726_ , 2024. 

- [8] H. Caesar, J. Uijlings, and V. Ferrari. Cocostuff: Thing and stuff classes in context. In _CVPR_ , 2018. 

- [9] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In _CVPR_ , pages 9650– 9660, 2021. 

- [10] X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, A. Kolesnikov, J. Puigcerver, N. Ding, K. Rong, H. Akbari, G. Mishra, L. Xue, A. Thapliyal, J. Bradbury, W. Kuo, M. Seyedhosseini, C. Jia, B. K. Ayan, C. Riquelme, A. Steiner, A. Angelova, X. Zhai, N. Houlsby, and R. Soricut. PaLI: A jointly-scaled multilingual language-image model. _arXiv:2209.06794_ , 2022. 

- [11] S. Cho, H. Shin, S. Hong, A. Arnab, P. H. Seo, and S. Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In _CVPR_ , pages 4113–4123, 2024. 

- [12] M. Dehghani, B. Mustafa, J. Djolonga, J. Heek, M. Minderer, M. Caron, A. Steiner, J. Puigcerver, R. Geirhos, I. M. Alabdulmohsin, et al. Patch n’pack: NaViT, a vision transformer for any aspect ratio and resolution. _NeurIPS_ , 2024. 

- [13] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In _CVPR_ , pages 248–255, 2009. 

- [14] J. Ding, N. Xue, G.-S. Xia, and D. Dai. Decoupling zero-shot semantic segmentation. In _CVPR_ , pages 11583–11592, 2022. 

- [15] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 

- [16] T. Evans, N. Parthasarathy, H. Merzic, and O. J. Henaff. Data curation via joint example selection further accelerates multimodal learning. In _NeurIPS Datasets and Benchmarks Track_ , 2024. 

- [17] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. _IJCV_ , 2010. 

- [18] L. Fan, D. Krishnan, P. Isola, D. Katabi, and Y. Tian. Improving clip training with language rewrites. _NeurIPS_ , pages 35544– 35575, 2023. 

- [19] A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. T. Toshev, and V. Shankar. Data filtering networks. In _ICLR_ , 2024. 

- [20] E. Fini, M. Shukor, X. Li, P. Dufter, M. Klein, D. Haldimann, S. Aitharaju, V. G. T. da Costa, L. Béthune, Z. Gan, A. T. Toshev, M. Eichner, M. Nabi, Y. Yang, J. M. Susskind, and A. El-Nouby. Multimodal autoregressive pre-training of large vision encoders. _arXiv:2411.14402_ , 2024. 

- [21] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. _NeurIPS_ , 36, 2024. 

- [22] Gemma Team. Gemma: Open models based on gemini research and technology. _arXiv:2403.08295_ , 2024. 

- [23] Gemma Team. Gemma 2: Improving open language models at a practical size. _arXiv:2408.00118_ , 2024. 

- [24] Google Cloud. Introduction to Cloud TPU. `https://cloud.google.com/ tpu/docs/intro-to-tpu` , 20xx. Accessed: 2024-07-04. 

13 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

- [25] A. Gupta, P. Dollar, and R. Girshick. Lvis: A dataset for large vocabulary instance segmentation. In _CVPR_ , pages 5356–5364, 2019. 

- [26] T.-Y. Hsu, C. L. Giles, and T.-H. Huang. Scicap: Generating captions for scientific figures. _arXiv:2110.11624_ , 2021. 

- [27] G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave, V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and L. Schmidt. OpenCLIP, 2021. 

- [28] C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li, and T. Duerig. Scaling up visual and visionlanguage representation learning with noisy text supervision. In _ICML_ , 2021. 

- [29] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg. ReferItGame: Referring to objects in photographs of natural scenes. In _EMNLP_ , Oct. 2014. 

- [30] W. Kuo, Y. Cui, X. Gu, A. Piergiovanni, and A. Angelova. Open-vocabulary object detection upon frozen vision and language models. In _ICLR_ , 2023. 

- [31] Z. Lai, H. Zhang, B. Zhang, W. Wu, H. Bai, A. Timofeev, X. Du, Z. Gan, J. Shan, C.-N. Chuah, Y. Yang, and M. Cao. VeCLIP: Improving clip training via visual-enriched captions. _arXiv:2310.07699_ , 2024. 

- [32] J. Li, D. Li, S. Savarese, and S. C. H. Hoi. BLIP-2: bootstrapping language-image pretraining with frozen image encoders and large language models. In _ICML_ , 2023. 

- [33] X. Li, Z. Wang, and C. Xie. Clipa-v2: Scaling clip training with 81.1% zero-shot imagenet accuracy within a $10,000 budget; an extra $4,000 unlocks 81.8% accuracy. _arXiv:2306.15658_ , 2023. 

- [34] T. Lin, M. Maire, S. J. Belongie, L. D. Bourdev, R. B. Girshick, J. Hays, P. Perona, D. Ramanan, P. Doll’a r, and C. L. Zitnick. Microsoft COCO: common objects in context. _arXiv:1405.0312_ , 2014. 

- [35] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In _NeurIPS_ , 2023. 

- [36] S. Long, S. Qin, D. Panteleev, A. Bissacco, Y. Fujii, and M. Raptis. ICDAR 2023 competition on hierarchical text detection and recognition. In _ICDAR_ , 2023. 

- [37] I. Loshchilov, F. Hutter, et al. Fixing weight decay regularization in adam. _arXiv preprint arXiv:1711.05101_ , 5, 2017. 

- [38] K.-K. Maninis, K. Chen, S. Ghosh, A. Karpur, K. Chen, Y. Xia, B. Cao, D. Salz, G. Han, J. Dlabal, et al. TIPS: Text-image pretraining with spatial awareness. In _ICLR_ , 2025. 

- [39] B. McKinzie, Z. Gan, J. Fauconnier, S. Dodge, B. Zhang, P. Dufter, D. Shah, X. Du, F. Peng, F. Weers, A. Belyi, H. Zhang, K. Singh, D. Kang, A. Jain, H. Hè, M. Schwarzer, T. Gunter, X. Kong, A. Zhang, J. Wang, C. Wang, N. Du, T. Lei, S. Wiseman, G. Yin, M. Lee, Z. Wang, R. Pang, P. Grasch, A. Toshev, and Y. Yang. MM1: methods, analysis & insights from multimodal LLM pretraining. _arXiv:2403.09611_ , 2024. 

- [40] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, et al. Simple open-vocabulary object detection. In _ECCV_ , pages 728–755, 2022. 

- [41] M. Minderer, A. A. Gritsenko, and N. Houlsby. Scaling open-vocabulary object detection. In _NeurIPS_ , 2023. 

- [42] S. Mindermann, J. M. Brauner, M. T. Razzak, M. Sharma, A. Kirsch, W. Xu, B. Höltgen, A. N. Gomez, A. Morisot, S. Farquhar, et al. Prioritized training on points that are learnable, worth learning, and not yet learnt. In _ICML_ , pages 15630–15649, 2022. 

- [43] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.W. Lee, S. Fidler, R. Urtasun, and A. Yuille. The role of context for object detection and semantic segmentation in the wild. In _CVPR_ , 2014. 

14 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

- [44] N. Mu, A. Kirillov, D. Wagner, and S. Xie. SLIP: Self-supervision meets languageimage pre-training. In _ECCV_ , pages 529– 544, 2022. 

- [45] M. F. Naeem, Y. Xian, X. Zhai, L. Hoyer, L. Van Gool, and F. Tombari. SILC: Improving vision language pretraining with selfdistillation. In _ECCV_ , pages 38–55, 2024. 

- [46] T. Nguyen, S. Y. Gadre, G. Ilharco, S. Oh, and L. Schmidt. Improving multimodal datasets with image captioning. _NeurIPS_ , 36, 2024. 

- [47] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _TMLR_ , 2024. 

- [48] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei. Kosmos-2: Grounding multimodal large language models to the world. _arXiv:2306.14824_ , 2023. 

- [49] A. Pouget, L. Beyer, E. Bugliarello, X. Wang, A. P. Steiner, X. Zhai, and I. Alabdulmohsin. No filter: Cultural and socioeconomic diversityin contrastive vision-language models. _arXiv:2405.13777_ , 2024. 

- [50] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [51] V. V. Ramaswamy, S. Y. Lin, D. Zhao, A. Adcock, L. van der Maaten, D. Ghadiyaram, and O. Russakovsky. Geode: a geographically diverse evaluation dataset for object recognition. _NeurIPS_ , 36, 2024. 

- [52] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. In _CVPR_ , pages 12179–12188, 2021. 

- [53] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? In _ICML_ , pages 5389– 5400, 2019. 

- [54] W. A. G. Rojas, S. Diamos, K. R. Kini, D. Kanter, V. J. Reddi, and C. Coleman. The dollar street dataset: Images representing the geographic and socioeconomic diversity of the world. In _NeurIPS Datasets and Benchmarks Track_ , 2022. 

- [55] O. Sidorov, R. Hu, M. Rohrbach, and A. Singh. TextCaps: A dataset for image captioning with reading comprehension. In _ECCV_ , 2020. 

- [56] A. Steiner, A. S. Pinto, M. Tschannen, D. Keysers, X. Wang, Y. Bitton, A. Gritsenko, M. Minderer, A. Sherbondy, S. Long, et al. Paligemma 2: A family of versatile vlms for transfer. _arXiv:2412.03555_ , 2024. 

- [57] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao. EVA-CLIP: Improved training techniques for clip at scale. _arXiv:2303.15389_ , 2023. 

- [58] A. V. Thapliyal, J. Pont Tuset, X. Chen, and R. Soricut. Crossmodal-3600: A massively multilingual multimodal evaluation dataset. In _EMNLP_ , 2022. 

- [59] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan, A. Wang, R. Fergus, Y. LeCun, and S. Xie. Cambrian-1: A Fully Open, VisionCentric Exploration of Multimodal LLMs. _arXiv:2406.16860_ , 2024. 

- [60] M. Tschannen, M. Kumar, A. Steiner, X. Zhai, N. Houlsby, and L. Beyer. Image captioners are scalable vision learners too. In _NeurIPS_ , 2023. 

- [61] V. Udandarao, N. Parthasarathy, M. F. Naeem, T. Evans, S. Albanie, F. Tombari, Y. Xian, A. Tonioni, and O. J. Hénaff. Active data curation effectively distills large-scale multimodal models. _arXiv:2411.18674_ , 2024. 

- [62] B. Wan, M. Tschannen, Y. Xian, F. Pavetic, I. Alabdulmohsin, X. Wang, A. S. Pinto, A. Steiner, L. Beyer, and X. Zhai. LocCa: Visual pretraining with location-aware captioners. In _NeurIPS_ , 2024. 

15 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

- [63] B. Wang, G. Li, X. Zhou, Z. Chen, T. Grossman, and Y. Li. Screen2words: Automatic mobile ui summarization with multimodal learning. In _Symposium on User Interface Software and Technology_ , 2021. 

- [64] Z. Wang, J. Yu, A. W. Yu, Z. Dai, Y. Tsvetkov, and Y. Cao. SimVLM: Simple visual language model pretraining with weak supervision. In _ICLR_ , 2022. 

- [65] T. Weyand, A. Araujo, B. Cao, and J. Sim. Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval. In _CVPR_ , pages 2575–2584, 2020. 

   - [73] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba. Scene parsing through ade20k dataset. In _CVPR_ , 2017. 

   - [74] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba. Semantic understanding of scenes through the ade20k dataset. _IJCV_ , 2019. 

   - [75] J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong. Image BERT pretraining with online tokenizer. In _ICLR_ , 2022. 

- [66] H. Xu, S. Xie, X. Tan, P.-Y. Huang, R. Howes, V. Sharma, S.-W. Li, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer. Demystifying clip data. In _ICLR_ , 2024. 

- [67] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu. CoCa: Contrastive captioners are image-text foundation models. _TMLR_ , 2022. 

- [68] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg. Modeling context in referring expressions. In _ECCV_ , pages 69–85, 2016. 

- [69] X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer. Scaling vision transformers. _CVPR_ , 2022. 

- [70] X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _CVPR_ , 2022. 

- [71] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In _ICCV_ , 2023. 

- [72] Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, P. Damania, B. Nguyen, G. Chauhan, Y. Hao, A. Mathews, and S. Li. Pytorch FSDP: experiences on scaling fully sharded data parallel. _VLDB_ , 2023. 

16 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

## **Appendix** 

## **A. Full PaliGemma results** 

||Large 224/256px<br>SigLIP<br>AIMv2<br>SigLIP2|So400m/14 224px<br>SigLIP<br>SigLIP2|So400m 384px<br>SigLIP<br>SigLIP2|
|---|---|---|---|
|AI2D<br>AOKVQA-DA (val)<br>AOKVQA-MC (val)<br>COCO-35L (avg34)<br>COCO-35L (en)<br>COCOcap<br>CountBenchQA<br>DocVQA (val)<br>GQA<br>InfoVQA (val)<br>NLVR2<br>NoCaps<br>OCR-VQA<br>OKVQA<br>RefCOCO (testA)<br>RefCOCO (testB)<br>RefCOCO (val)<br>RefCOCO+ (testA)<br>RefCOCO+ (testB)<br>RefCOCO+ (val)<br>RefCOCOg (test)<br>RefCOCOg (val)<br>ST-VQA (val)<br>SciCap<br>ScienceQA<br>Screen2Words<br>TallyQA (complex)<br>TallyQA (simple)<br>TextCaps<br>TextVQA (val)<br>VQAv2 (minival)<br>VizWizVQA (val)<br>WidgetCap<br>XM3600 (avg35)<br>XM3600 (en)|75.2<br>73.2<br>75.9<br>60.3<br>62.3<br>61.7<br>78.3<br>78.4<br>77.6<br>109.9<br>111.4<br>112.2<br>136.7<br>138.3<br>139.4<br>138.6<br>139.9<br>141.3<br>75.3<br>83.1<br>82.2<br>33.0<br>32.3<br>35.4<br>65.2<br>65.6<br>66.1<br>25.3<br>25.1<br>26.3<br>90.7<br>91.3<br>91.1<br>117.7<br>121.7<br>120.3<br>70.6<br>71.8<br>72.5<br>62.4<br>62.7<br>63.3<br>71.0<br>71.9<br>74.3<br>66.0<br>67.8<br>70.3<br>68.7<br>69.5<br>72.4<br>67.5<br>69.0<br>70.8<br>59.6<br>61.5<br>63.3<br>63.6<br>65.1<br>67.6<br>63.9<br>65.4<br>67.5<br>63.3<br>64.3<br>66.8<br>54.0<br>53.9<br>59.8<br>161.1<br>156.4<br>165.5<br>96.1<br>96.1<br>96.2<br>108.7<br>106.9<br>114.3<br>67.6<br>69.4<br>69.3<br>79.9<br>81.0<br>82.0<br>116.5<br>116.8<br>126.1<br>51.9<br>53.9<br>57.3<br>81.5<br>82.1<br>82.1<br>74.4<br>74.4<br>76.0<br>132.8<br>133.0<br>139.1<br>39.0<br>39.6<br>39.7<br>77.7<br>78.0<br>79.1|75.3<br>74.8<br>62.0<br>62.8<br>79.0<br>80.5<br>111.9<br>113.2<br>139.0<br>139.4<br>141.4<br>142.7<br>78.2<br>84.7<br>34.3<br>35.9<br>65.5<br>65.7<br>25.1<br>26.0<br>91.0<br>91.4<br>120.1<br>120.9<br>71.3<br>72.7<br>63.1<br>63.4<br>72.4<br>74.5<br>67.5<br>70.5<br>69.9<br>72.5<br>69.0<br>71.4<br>60.8<br>63.3<br>64.9<br>67.8<br>64.7<br>67.9<br>64.5<br>67.3<br>56.7<br>60.1<br>162.3<br>161.8<br>95.4<br>96.3<br>111.3<br>110.6<br>68.4<br>70.0<br>80.4<br>82.2<br>121.7<br>123.8<br>54.5<br>59.4<br>81.9<br>82.8<br>75.5<br>76.0<br>134.4<br>142.0<br>39.8<br>40.1<br>77.8<br>79.2|76.7<br>78.3<br>64.9<br>64.7<br>82.5<br>83.1<br>113.6<br>114.8<br>140.3<br>141.1<br>142.2<br>143.8<br>80.8<br>83.9<br>62.7<br>65.9<br>67.0<br>67.8<br>34.7<br>37.1<br>91.7<br>91.8<br>120.8<br>121.9<br>74.4<br>75.2<br>63.7<br>64.5<br>76.6<br>78.2<br>71.4<br>74.5<br>74.3<br>76.1<br>74.1<br>75.9<br>65.4<br>67.6<br>70.0<br>72.0<br>69.9<br>72.1<br>69.5<br>71.7<br>75.0<br>77.3<br>177.2<br>179.3<br>96.2<br>96.1<br>115.3<br>116.1<br>71.0<br>72.5<br>83.5<br>85.4<br>145.0<br>150.9<br>69.7<br>74.0<br>84.3<br>85.2<br>76.8<br>77.6<br>147.0<br>151.1<br>40.8<br>41.1<br>80.0<br>81.0|



Table 6 | The first three columns compare Large-sized models with 256 tokens each (that’s 224px for the AIMv2 model with patch size 14, and 256px for the SigLIP models with patch size 16). The last four columns compare So400M-sized SigLIP models with patch size 14 at two different resolutions (and hence tokens). Same data as in Figure 4. 

17 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

## **B. Full NaFlex results** 

|ViT|Seq. Model|ImageNet-1k<br>COCO R@1<br>TC R@1<br>HT R@1<br>SC R@1<br>S2W R@1<br>val<br>v2<br>ReaL ObjNet T→I<br>I→T<br>T→I I→T T→I I→T T→I I→T T→I I→T|
|---|---|---|
|B/16|64 SigLIP 2(NaF.) 71.2 63.2 78.3<br>62.1<br>43.6<br>60.4<br>30.4 57.5<br>3.4<br>6.4<br>5.2<br>4.0<br>6.4<br>11.0||
||144 SigLIP 2(NaF.) 76.2 69.4 82.9<br>70.2<br>49.0<br>65.7<br>36.5 65.8<br>5.7<br>10.3 13.5 11.8 13.9 25.4||
||196 SigLIP 2<br>78.2 71.4 84.8<br>73.6<br>52.1<br>68.9<br>38.9 68.0<br>5.5<br>9.5<br>13.3 10.9 10.8 18.7||
||256 SigLIP 2<br>79.1 72.5 85.4<br>74.5<br>53.2<br>69.7<br>40.5 69.4<br>6.1<br>9.8<br>17.1 14.2 12.9 22.9<br>SigLIP 2(NaF.) 78.5 71.9 84.6<br>74.6<br>51.1<br>67.3<br>39.5 69.0<br>7.4<br>12.9 19.7 17.1 14.8 26.6||
||576 SigLIP 2<br>80.6 73.8 86.2<br>77.1<br>54.6<br>71.4<br>43.6 73.0<br>7.5<br>12.0 23.3 19.4 14.1 24.8<br>SigLIP 2(NaF.) 80.0 73.1 85.6<br>76.4<br>52.5<br>69.1<br>41.6 71.8<br>8.7<br>14.1 24.3 21.0 15.3 26.7||
||676 SigLIP 2(NaF.) 80.1 73.5 85.7<br>76.5<br>52.9<br>68.6<br>41.8 73.0<br>8.8<br>13.9 24.3 21.4 15.2 26.2||
||784 SigLIP 2(NaF.) 80.2 73.5 85.9<br>76.9<br>53.1<br>68.8<br>42.5 72.9<br>8.7<br>14.0 24.8 21.5 15.2 26.4||
||900 SigLIP 2(NaF.) 80.3 73.6 85.9<br>76.6<br>52.9<br>69.2<br>42.3 72.6<br>8.6<br>15.0 24.8 21.6 15.0 25.8||
||1024 SigLIP 2<br>81.2 74.5 86.7<br>77.8<br>55.2<br>71.2<br>44.7 74.7<br>8.1<br>14.6 25.2 20.7 14.5 25.3<br>SigLIP 2(NaF.) 80.4 73.5 85.9<br>76.6<br>52.9<br>68.9<br>42.5 73.2<br>9.1<br>14.4 25.1 21.5 14.9 26.4||
|So/16|64 SigLIP 2(NaF.) 78.5 71.0 84.2<br>73.8<br>49.6<br>67.4<br>37.0 65.5<br>5.6<br>10.3 11.8 10.9 12.1 21.4||
||144 SigLIP 2(NaF.) 81.8 75.2 86.7<br>79.8<br>53.4<br>70.4<br>42.8 71.0<br>8.0<br>14.6 22.2 23.1 17.1 29.0||
||256 SigLIP 2<br>83.4 77.8 87.7<br>84.8<br>55.4<br>71.5<br>44.8 72.9<br>7.9<br>13.9 29.7 28.8 17.4 28.7<br>SigLIP 2(NaF.) 83.5 77.5 87.7<br>83.8<br>55.1<br>71.2<br>44.9 73.6<br>9.2<br>15.7 29.8 29.2 17.5 29.2||
||576 SigLIP 2<br>84.1 78.4 88.1<br>85.8<br>56.0<br>71.2<br>47.0 74.9<br>9.7<br>16.3 34.5 32.4 17.8 28.0<br>SigLIP 2(NaF.) 84.1 78.6 88.0<br>85.7<br>55.9<br>71.4<br>46.5 75.1 11.3 18.4 32.9 32.0 17.7 28.8||
||676 SigLIP 2(NaF.) 84.2 78.5 88.0<br>85.7<br>55.8<br>71.7<br>46.9 74.9 11.3 18.5 33.3 32.2 17.7 29.8||
||784 SigLIP 2(NaF.) 84.3 78.6 88.0<br>85.9<br>55.9<br>71.3<br>46.7 74.9 11.5 18.5 33.0 32.3 17.6 29.5||
||900 SigLIP 2(NaF.) 84.3 78.6 88.1<br>85.8<br>55.8<br>71.2<br>46.8 75.4 11.7 18.5 32.9 32.5 17.7 29.4||
||1024 SigLIP 2<br>84.3 79.1 88.1<br>86.2<br>56.0<br>71.3<br>47.3 76.0 10.3 18.3 35.9 33.5 17.9 28.1<br>SigLIP 2 (NaF.) 84.4 78.8 88.1<br>85.8<br>55.8<br>71.0<br>46.9 74.9 11.7 18.4 32.6 32.4 17.8 29.4||



Table 7 | Comparing the NaFlex (supporting native aspect ratio and variable sequence length (Seq.)) and the standard square-input SigLIP variants which use a separate checkpoint per sequence length. Numerical data corresponding to the plots in Fig. 3. TC: TextCaps, HT: HierText, SC: SciCap, S2W: Screen2Words. 

18 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

## **C. Full cultural diversity and fairness results** 

|ViT|Res.<br>Model|10-shot<br>Dollar Street<br>GeoDE (country)<br>GeoDE (region)|0-shot<br>Dollar Street<br>GLDv2<br>GeoDE|
|---|---|---|---|
|B/32|256<br>SigLIP 2|13.1<br>13.9<br>29.3|50.5<br>44.7<br>90.6|
|B/16|224<br>SigLIP<br>SigLIP 2|13.8<br>12.7<br>27.3<br>16.2<br>20.0<br>34.9|50.1<br>48.5<br>92.4<br>53.4<br>50.8<br>92.9|
||256<br>SigLIP<br>SigLIP 2|15.0<br>13.3<br>29.3<br>17.7<br>22.7<br>36.3|50.3<br>47.7<br>92.8<br>54.2<br>52.5<br>93.3|
||384<br>SigLIP<br>SigLIP 2|16.1<br>16.4<br>31.5<br>19.8<br>25.6<br>41.4|51.5<br>51.9<br>93.6<br>54.8<br>55.2<br>93.9|
||512<br>SigLIP<br>SigLIP 2|16.6<br>17.7<br>32.3<br>21.7<br>28.2<br>43.1|51.3<br>53.1<br>94.1<br>54.9<br>57.6<br>94.2|
|L/16|256<br>SigLIP<br>SigLIP 2|18.8<br>22.1<br>36.2<br>26.8<br>34.5<br>44.4|52.1<br>56.7<br>93.6<br>55.2<br>64.5<br>94.9|
||384<br>SigLIP<br>SigLIP 2|22.8<br>26.0<br>41.7<br>30.4<br>39.3<br>48.0|52.9<br>60.5<br>94.3<br>55.4<br>66.1<br>95.1|
||512<br>SigLIP 2|32.5<br>42.5<br>50.6|55.2<br>67.6<br>95.3|
|So400m/14|224<br>SigLIP<br>SigLIP 2|26.6<br>31.9<br>45.8<br>31.9<br>38.1<br>49.1|55.1<br>74.1<br>94.7<br>55.4<br>65.6<br>94.8|
||384<br>SigLIP<br>SigLIP 2|32.1<br>36.5<br>51.6<br>38.3<br>45.2<br>56.1|56.3<br>71.7<br>94.9<br>56.6<br>68.6<br>95.2|
|So400m/16|256<br>SigLIP 2<br>mSigLIP|33.2<br>39.8<br>50.9<br>27.1<br>33.3<br>48.5|55.8<br>66.7<br>95.0<br>54.2<br>57.5<br>94.3|
||384<br>SigLIP 2|38.2<br>44.1<br>54.4|56.5<br>67.8<br>95.3|
||512<br>SigLIP 2|40.8<br>47.6<br>58.6|56.6<br>69.2<br>95.3|
|g-opt/16|256<br>SigLIP 2|37.6<br>46.6<br>54.0|56.9<br>71.2<br>95.4|
||384<br>SigLIP 2|44.5<br>52.0<br>58.7|57.2<br>72.2<br>95.7|



Table 8 | 10-shot and 0-shot accuracy for geographically diverse object classification tasks (Dollar Street, GeoDE), as well as geolocalization (GeoDE country/region) and landmark localization (GLDv2) tasks. SigLIP 2 consistently outperforms SigLIP on most benchmarks. 

19 

SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

|ViT|Res.<br>Model<br>Disparity<br>Rep. bias|
|---|---|
|B/32|256<br>SigLIP 2<br>33.3<br>16.6|
|B/16|224<br>SigLIP<br>31.2<br>36.6<br>SigLIP 2<br>31.0<br>17.2|
||256<br>SigLIP<br>30.2<br>35.6<br>SigLIP 2<br>29.7<br>19.4|
||384<br>SigLIP<br>30.9<br>35.8<br>SigLIP 2<br>30.6<br>18.0|
||512<br>SigLIP<br>31.5<br>35.4<br>SigLIP 2<br>30.8<br>20.0|
|L/16|256<br>SigLIP<br>32.0<br>35.5<br>SigLIP 2<br>31.1<br>7.3|
||384<br>SigLIP<br>32.0<br>34.8<br>SigLIP 2<br>30.4<br>6.6|
||512<br>SigLIP 2<br>29.2<br>6.8|
|So400m/14|224<br>SigLIP<br>30.5<br>33.3<br>SigLIP 2<br>29.7<br>7.4|
||384<br>SigLIP<br>29.2<br>33.9<br>SigLIP 2<br>28.1<br>7.5|
|So400m/16|256<br>SigLIP 2<br>28.4<br>7.2<br>mSigLIP<br>31.6<br>37.3|
||384<br>SigLIP 2<br>29.0<br>11.0|
||512<br>SigLIP 2<br>28.2<br>10.8|
|g-opt/16|256<br>SigLIP 2<br>28.1<br>7.9|
||384<br>SigLIP 2<br>28.3<br>4.9|



Table 9 | Disparity: Corresponds to the maximum difference in 0-shot accuracy on Dollar Street when disaggregating the accuracy by income level: We observe that SigLIP 2 slightly reduces the performance disparity. Rep. bias: Representation bias; lower values are better. SigLIP2, which is trained on de-biased data, exhibits significantly reduced representation bias than its predecessor. In addition, larger models are better than smaller models, in agreement with the earlier findings in [2]. 

20 

