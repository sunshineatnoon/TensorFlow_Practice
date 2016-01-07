#include <stdio.h>
#include <stdlib.h>

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    CROP,
    MAXPOOL,
    AVGPOOL,
    SOFTMAX,
    COST
} LAYER_TYPE;

typedef struct layer{
  int size;//filter size
  int c;//channels, number of input feature maps
  int n;// number of filters in this layer
  int h;
  int w;
  int input_size;
  int output_size;
  float *biases;
  float *weights;
  LAYER_TYPE type;
}layer;

typedef struct network{
  int layer_number;
  struct layer *layers;

}network;

layer make_layer(int size,int c,int n,int h,int w,LAYER_TYPE type){
  layer l = {0};
  l.size = size;
  l.c = c;
  l.n = n;
  l.h = h;
  l.w = w;
  l.type = type;
  l.biases = calloc(n, sizeof(float));
  l.weights = calloc(c*n*size*size, sizeof(float));
  return l;
}

typedef layer connected_layer;
connected_layer make_connected_layer(int input_size,int output_size){
  connected_layer l = {0};
  l.input_size = input_size;
  l.output_size = output_size;
  l.type = CONNECTED;
  l.weights = calloc(output_size*input_size, sizeof(float));
  l.biases = calloc(output_size, sizeof(float));
  return l;
}
void main(){
  FILE *file;

  //construct Google Net
  struct network GoogleNet = {
    .layer_number = 29
  };
  GoogleNet.layers = (layer*)malloc(sizeof(layer)*29);

  GoogleNet.layers[0] = make_layer(0,0,0,0,0,CROP);
  GoogleNet.layers[1] = make_layer(7,3,64,224,224,CONVOLUTIONAL);
  GoogleNet.layers[2] = make_layer(0,0,0,0,0,MAXPOOL);
  GoogleNet.layers[3] = make_layer(3,64,192,56,56,CONVOLUTIONAL);
  GoogleNet.layers[4] = make_layer(0,0,0,0,0,MAXPOOL);
  GoogleNet.layers[5] = make_layer(1,192,128,28,28,CONVOLUTIONAL);
  GoogleNet.layers[6] = make_layer(3,128,256,28,28,CONVOLUTIONAL);
  GoogleNet.layers[7] = make_layer(1,256,256,28,28,CONVOLUTIONAL);
  GoogleNet.layers[8] = make_layer(3,256,512,28,28,CONVOLUTIONAL);
  GoogleNet.layers[9] = make_layer(0,0,0,0,0,MAXPOOL);
  GoogleNet.layers[10] = make_layer(1,512,256,14,14,CONVOLUTIONAL);
  GoogleNet.layers[11] = make_layer(3,256,512,14,14,CONVOLUTIONAL);
  GoogleNet.layers[12] = make_layer(1,512,256,14,14,CONVOLUTIONAL);
  GoogleNet.layers[13] = make_layer(3,256,512,14,14,CONVOLUTIONAL);
  GoogleNet.layers[14] = make_layer(1,512,256,14,14,CONVOLUTIONAL);
  GoogleNet.layers[15] = make_layer(3,256,512,14,14,CONVOLUTIONAL);
  GoogleNet.layers[16] = make_layer(1,512,256,14,14,CONVOLUTIONAL);
  GoogleNet.layers[17] = make_layer(3,256,512,14,14,CONVOLUTIONAL);
  GoogleNet.layers[18] = make_layer(1,512,512,14,14,CONVOLUTIONAL);
  GoogleNet.layers[19] = make_layer(3,512,1024,14,14,CONVOLUTIONAL);
  GoogleNet.layers[20] = make_layer(0,0,0,0,0,MAXPOOL);
  GoogleNet.layers[21] = make_layer(1,1024,512,7,7,CONVOLUTIONAL);
  GoogleNet.layers[22] = make_layer(3,512,1024,7,7,CONVOLUTIONAL);
  GoogleNet.layers[23] = make_layer(1,1024,512,7,7,CONVOLUTIONAL);
  GoogleNet.layers[24] = make_layer(3,512,1024,7,7,CONVOLUTIONAL);
  GoogleNet.layers[25] = make_layer(0,0,0,0,0,AVGPOOL);
  GoogleNet.layers[26] = make_connected_layer(1024,1000);
  GoogleNet.layers[27] = make_layer(0,0,0,0,0,SOFTMAX);
  GoogleNet.layers[28] = make_layer(0,0,0,0,0,COST);

  if((file = fopen("extraction.weights","r")) != NULL){
    float learning_rate,momentum,decay;
    int seen;
    fread(&learning_rate, sizeof(float), 1, file);
    fread(&momentum, sizeof(float), 1, file);
    fread(&decay, sizeof(float), 1, file);
    fread(&seen, sizeof(int), 1, file);
    printf("The three numbers read from the file are: %f,%f,%f,%d\n",learning_rate,momentum,decay,seen);

    int i;
    for(i = 0;i < GoogleNet.layer_number;i++){
      layer l = GoogleNet.layers[i];
      if(l.type == CONVOLUTIONAL){
        int num = l.n*l.c*l.size*l.size;
        fread(l.biases, sizeof(float), l.n, file);
        fread(l.weights, sizeof(float), num, file);
      }else if(l.type == CONNECTED){
        fread(l.biases, sizeof(float), l.output_size, file);
        fread(l.weights, sizeof(float), l.output_size*l.input_size, file);
      }
      printf("%d. (%d,%d,%d,%d,%d)\n",i+1,l.size,l.n,l.c,l.h,l.w);
    }
  }
  //If I read these weights correctly, then when I write them back, the weights should work well as before,let's try!
  
}
