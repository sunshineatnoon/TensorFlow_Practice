#include <stdio.h>
#include <stdlib.h>

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    CROP,
    MAXPOOL,
    AVGPOOL,
    SOFTMAX
} LAYER_TYPE;

typedef struct layer{
  int size;//filter size
  int c;//channels, number of input feature maps
  int n;// number of filters in this layer
  int input_size;
  int output_size;
  float *biases;
  float *weights;
  LAYER_TYPE type;
}layer;

typedef struct network{
  int layer_numbers;
  struct layer *layers;

}network;

typedef struct layer convolutional_layer;
convolutional_layer make_convolutional_layer(int size,int c,int n,int input_size,int output_size,LAYER_TYPE type){
  convolutional_layer l = {0};
  l.size = size;
  l.c = c;
  l.n = n;
  l.input_size = input_size;
  l.output_size = output_size;
  l.type = CONVOLUTIONAL;
  l.biases = calloc(n, sizeof(float));
  l.weights = calloc(c*n*size*size, sizeof(float));
  return l;
}

void main(){
  FILE *file;

  //construct Google Net
  struct network GoogleNet = {
    .layer_numbers = 30
  };
  GoogleNet.layers = (layer*)malloc(sizeof(layer)*30);
  //manually define the 30 layers GoogleNet
  GoogleNet.layers[0] = make_convolutional_layer(0,0,0,0,0,CROP);

  if((file = fopen("extraction.weights","r")) != NULL){
    float learning_rate,momentum,decay;
    int seen;
    fread(&learning_rate, sizeof(float), 1, file);
    fread(&momentum, sizeof(float), 1, file);
    fread(&decay, sizeof(float), 1, file);
    fread(&seen, sizeof(int), 1, file);
    printf("The three numbers read from the file are: %f,%f,%f,%d\n",learning_rate,momentum,decay,seen);
    fclose(file);
  }
  int i;
  for(i = 0;i < 1;i++){
    printf("(%d)\n",GoogleNet.layers[i].size);
  }
}
