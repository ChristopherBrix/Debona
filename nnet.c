/*
   ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang
 ** This file is part of the Neurify project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include "nnet.h"


int PROPERTY = 5;
char *LOG_FILE = "logs/log.txt";
float INF = 1;

int ERR_NODE=5000;

int NORM_INPUT=1;

int CHECK_ADV_MODE = 0;

struct timeval start,finish,last_finish;
//FILE *fp;

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
struct NNet *load_conv_network(const char* filename, int img)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");
    
    if (fstream == NULL)
    {
        printf("Wrong network!\n");
        exit(1);
    }
    //Initialize variables
    int bufferSize = 650000;
    char *buffer = (char*)SAFEMALLOC(sizeof(char)*bufferSize);
    char *record, *line;

    struct NNet *nnet = (struct NNet*)SAFEMALLOC(sizeof(struct NNet));

    //memset(nnet, 0, sizeof(struct NNet));
    //Read int parameters of neural network

    line=fgets(buffer,bufferSize,fstream);
    while (strstr(line, "//")!=NULL)
        line=fgets(buffer,bufferSize,fstream); //skip header lines
    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = (int*)SAFEMALLOC(sizeof(int)*(nnet->numLayers+1+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }
    nnet->layerSizes[nnet->numLayers + 1] = 1;
    //Load Min and Max values of inputs
    nnet->min = MIN_PIXEL;
    nnet->max = MAX_PIXEL;
    
    nnet->layerTypes = (int*)SAFEMALLOC(sizeof(int)*nnet->numLayers);
    nnet->convLayersNum = 0;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<nnet->numLayers; i++)
    {
        nnet->layerTypes[i] = atoi(record);
        if(nnet->layerTypes[i]==1){
            nnet->convLayersNum++;
        }
        record = strtok(NULL,",\n");
    }
    //initial convlayer parameters
    nnet->convLayer = (int**)SAFEMALLOC(sizeof(int *)*nnet->convLayersNum);
    for(int i = 0; i < nnet->convLayersNum; i++){
        nnet->convLayer[i] = (int*)SAFEMALLOC(sizeof(int)*5);
    }

    for(int cl=0;cl<nnet->convLayersNum;cl++){
        line = fgets(buffer,bufferSize,fstream);
        record = strtok(line,",\n");
        for (int i = 0; i<5; i++){
            nnet->convLayer[cl][i] = atoi(record);
            //printf("%d,", nnet->convLayer[cl][i]);
            record = strtok(NULL,",\n");
        }
        //printf("\n");
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only one number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = (float****)SAFEMALLOC(sizeof(float *)*(nnet->numLayers));
    for (int layer = 0; layer<nnet->numLayers; layer++){
        if(nnet->layerTypes[layer]==0){
            nnet->matrix[layer] = (float***)SAFEMALLOC(sizeof(float *)*2);
            nnet->matrix[layer][0] = (float**)SAFEMALLOC(sizeof(float *)*nnet->layerSizes[layer+1]);
            nnet->matrix[layer][1] = (float**)SAFEMALLOC(sizeof(float *)*nnet->layerSizes[layer+1]);
            for (int row = 0; row < nnet->layerSizes[layer+1]; row++){
                nnet->matrix[layer][0][row] = (float*)SAFEMALLOC(sizeof(float)*nnet->layerSizes[layer]);
                nnet->matrix[layer][1][row] = (float*)SAFEMALLOC(sizeof(float));
            }
        }
    }

    nnet->conv_matrix = (float****)SAFEMALLOC(sizeof(float *)*nnet->convLayersNum);
    for(int layer = 0; layer < nnet->convLayersNum; layer++){
        int out_channel = nnet->convLayer[layer][0];
        int in_channel = nnet->convLayer[layer][1];
        int kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
        nnet->conv_matrix[layer]=(float***)SAFEMALLOC(sizeof(float*)*out_channel);
        for(int oc=0;oc<out_channel;oc++){
            nnet->conv_matrix[layer][oc] = (float**)SAFEMALLOC(sizeof(float*)*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                nnet->conv_matrix[layer][oc][ic] = (float*)SAFEMALLOC(sizeof(float)*kernel_size);
            }

        }
    }

    nnet->conv_bias = (float**)SAFEMALLOC(sizeof(float*)*nnet->convLayersNum);
    for(int layer = 0; layer < nnet->convLayersNum; layer++){
        int out_channel = nnet->convLayer[layer][0];
        nnet->conv_bias[layer] = (float*)SAFEMALLOC(sizeof(float)*out_channel);
    }
    
    int layer = 0;
    int param = 0;
    int i=0;
    int j=0;
    char *tmpptr=NULL;

    int oc=0, ic=0, kernel=0;
    int out_channel=0,kernel_size=0;

    //Read in parameters and put them in the matrix
    float w = 0.0;
    while((line=fgets(buffer,bufferSize,fstream))!=NULL){
        if(nnet->layerTypes[layer]==1){
            out_channel = nnet->convLayer[layer][0];
            kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
            if(oc>=out_channel){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                    if(nnet->layerTypes[layer]==1){
                        out_channel = nnet->convLayer[layer][0];
                        kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
                    }                    
                }
                oc=0;
                ic=0;
                kernel=0;
            }
        }
        else{
            if(i>=nnet->layerSizes[layer+1]){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                }
                i=0;
                j=0;
            }
        }

        if(nnet->layerTypes[layer]==1){
            if(param==0){
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {   

                    w = (float)atof(record);
                    nnet->conv_matrix[layer][oc][ic][kernel] = w;
                    kernel++;
                    if(kernel==kernel_size){
                        kernel = 0;
                        ic++;
                    }
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                kernel=0;
                ic=0;
                oc++;
            }
            else{
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {   

                    w = (float)atof(record);
                    nnet->conv_bias[layer][oc] = w;
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                oc++;
            }
        }
        else{
            record = strtok_r(line,",\n", &tmpptr);
            while(record != NULL)
            {   
                w = (float)atof(record);
                nnet->matrix[layer][param][i][j] = w;                
                j++;
                record = strtok_r(NULL, ",\n", &tmpptr);
            }
            tmpptr=NULL;
            j=0;
            i++;
        }            
    }
    //printf("load matrix done\n");
    
    float input_prev[nnet->inputSize];
    struct Matrix input_prev_matrix = {input_prev, 1, nnet->inputSize};
    float o[nnet->outputSize];
    struct Matrix output = {o, nnet->outputSize, 1};
    //printf("start load inputs\n");
    load_inputs(img, nnet->inputSize, input_prev);
    //printf("load inputs done\n");
    if(NORM_INPUT){
        normalize_input(nnet, &input_prev_matrix);
    }
    //printf("normalize_input done\n");
    evaluate_conv(nnet, &input_prev_matrix, &output);
    
    float largest = output.data[0];
    nnet->target = 0;
    for(int o=1;o<nnet->outputSize;o++){
        if(output.data[o]>largest){
            largest = output.data[o];
            nnet->target = o;
        }
    }

    struct Matrix *weights_low = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *weights_low_gtzero = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *weights_up = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_low = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_low_gtzero = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_up = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));

    for(int layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==1) continue;
        weights_low[layer].row = nnet->layerSizes[layer];
        weights_low_gtzero[layer].row = nnet->layerSizes[layer];
        weights_up[layer].row = nnet->layerSizes[layer];
        weights_low[layer].col = nnet->layerSizes[layer+1];
        weights_low_gtzero[layer].col = nnet->layerSizes[layer+1];
        weights_up[layer].col = nnet->layerSizes[layer+1];
        weights_low[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_low[layer].row * weights_low[layer].col);
        weights_low_gtzero[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_low_gtzero[layer].row * weights_low_gtzero[layer].col);
        weights_up[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_up[layer].row * weights_up[layer].col);
        
        int n=0;
        for(int i=0;i<weights_low[layer].col;i++){
            for(int j=0;j<weights_low[layer].row;j++){
                float w = nnet->matrix[layer][0][i][j];
                weights_low[layer].data[n] = w;
                weights_low_gtzero[layer].data[n] = w;
                weights_up[layer].data[n] = w;
                n++;
            }
        }
        bias_low[layer].col = nnet->layerSizes[layer+1];
        bias_low_gtzero[layer].col = nnet->layerSizes[layer+1];
        bias_up[layer].col = nnet->layerSizes[layer+1];
        bias_low[layer].row = (float)1;
        bias_low_gtzero[layer].row = (float)1;
        bias_up[layer].row = (float)1;
        bias_low[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_low[layer].col);
        bias_low_gtzero[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_low_gtzero[layer].col);
        bias_up[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_up[layer].col);
        for(int i=0;i<bias_low[layer].col;i++){
            bias_low[layer].data[i] = nnet->matrix[layer][1][i][0];
            bias_low_gtzero[layer].data[i] = nnet->matrix[layer][1][i][0];
            bias_up[layer].data[i] = nnet->matrix[layer][1][i][0];
        }
    } 
    weights_low[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_low_gtzero[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_low_gtzero[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_up[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_low[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_low[nnet->numLayers].row * weights_low[nnet->numLayers].col);
    memset(weights_low[nnet->numLayers].data, 0, sizeof(float)*weights_low[nnet->numLayers].row * weights_low[nnet->numLayers].col);
    weights_low_gtzero[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_low_gtzero[nnet->numLayers].row * weights_low_gtzero[nnet->numLayers].col);
    memset(weights_low_gtzero[nnet->numLayers].data, 0, sizeof(float)*weights_low_gtzero[nnet->numLayers].row * weights_low_gtzero[nnet->numLayers].col);
    weights_up[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_up[nnet->numLayers].row * weights_up[nnet->numLayers].col);
    memset(weights_up[nnet->numLayers].data, 0, sizeof(float)*weights_up[nnet->numLayers].row * weights_up[nnet->numLayers].col);
    bias_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_low_gtzero[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_low[nnet->numLayers].row = (float)1;
    bias_low_gtzero[nnet->numLayers].row = (float)1;
    bias_up[nnet->numLayers].row = (float)1;
    bias_low[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_low[nnet->numLayers].col);
    memset(bias_low[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);
    bias_low_gtzero[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_low_gtzero[nnet->numLayers].col);
    memset(bias_low_gtzero[nnet->numLayers].data, 0, sizeof(float)*bias_low_gtzero[nnet->numLayers].col);
    bias_up[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_up[nnet->numLayers].col);
    memset(bias_up[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);

    nnet->weights_low = weights_low;
    nnet->weights_low_gtzero = weights_low_gtzero;
    nnet->weights_up = weights_up;
    nnet->bias_low = bias_low;
    nnet->bias_low_gtzero = bias_low_gtzero;
    nnet->bias_up = bias_up;

    free(buffer);
    fclose(fstream);
    
    nnet->cache_equation_low = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_equation_up = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_bias_low = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_bias_up = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_valid = false;

    
    nnet->cache_equation_low_gtzero = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_equation_up_gtzero = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_bias_low_gtzero = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_bias_up_gtzero = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_valid_gtzero = false;

    nnet->is_duplicate = false;

    return nnet;
}

struct NNet *duplicate_conv_network(struct NNet *orig_nnet)
{
    struct NNet *nnet = (struct NNet*)SAFEMALLOC(sizeof(struct NNet));

    nnet->numLayers = orig_nnet->numLayers;
    nnet->inputSize = orig_nnet->inputSize;
    nnet->outputSize = orig_nnet->outputSize;
    nnet->maxLayerSize = orig_nnet->maxLayerSize;

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = orig_nnet->layerSizes;

    //Load Min and Max values of inputs
    nnet->min = MIN_PIXEL;
    nnet->max = MAX_PIXEL;
    
    nnet->layerTypes = orig_nnet->layerTypes;
    nnet->convLayersNum = orig_nnet->convLayersNum;

    //initial convlayer parameters
    nnet->convLayer = orig_nnet->convLayer;

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only one number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = orig_nnet->matrix;

    nnet->conv_matrix = orig_nnet->conv_matrix;

    nnet->conv_bias = orig_nnet->conv_bias;

    nnet->target = orig_nnet->target;

    // Dont't copy weights and biases from orig_net! Those may have been
    // modified by ReLU relax operations
    struct Matrix *weights_low = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *weights_low_gtzero = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *weights_up = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_low = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_low_gtzero = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));
    struct Matrix *bias_up = SAFEMALLOC((nnet->numLayers+1)*sizeof(struct Matrix));

    for(int layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==1) continue;
        weights_low[layer].row = nnet->layerSizes[layer];
        weights_low_gtzero[layer].row = nnet->layerSizes[layer];
        weights_up[layer].row = nnet->layerSizes[layer];
        weights_low[layer].col = nnet->layerSizes[layer+1];
        weights_low_gtzero[layer].col = nnet->layerSizes[layer+1];
        weights_up[layer].col = nnet->layerSizes[layer+1];
        weights_low[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_low[layer].row * weights_low[layer].col);
        weights_low_gtzero[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_low_gtzero[layer].row * weights_low_gtzero[layer].col);
        weights_up[layer].data =\
                    (float*)SAFEMALLOC(sizeof(float)*weights_up[layer].row * weights_up[layer].col);
        
        int n=0;
        for(int i=0;i<weights_low[layer].col;i++){
            for(int j=0;j<weights_low[layer].row;j++){
                float w = nnet->matrix[layer][0][i][j];
                weights_low[layer].data[n] = w;
                weights_low_gtzero[layer].data[n] = w;
                weights_up[layer].data[n] = w;
                n++;
            }
        }
        bias_low[layer].col = nnet->layerSizes[layer+1];
        bias_low_gtzero[layer].col = nnet->layerSizes[layer+1];
        bias_up[layer].col = nnet->layerSizes[layer+1];
        bias_low[layer].row = (float)1;
        bias_low_gtzero[layer].row = (float)1;
        bias_up[layer].row = (float)1;
        bias_low[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_low[layer].col);
        bias_low_gtzero[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_low_gtzero[layer].col);
        bias_up[layer].data = (float*)SAFEMALLOC(sizeof(float)*bias_up[layer].col);
        for(int i=0;i<bias_low[layer].col;i++){
            bias_low[layer].data[i] = nnet->matrix[layer][1][i][0];
            bias_low_gtzero[layer].data[i] = nnet->matrix[layer][1][i][0];
            bias_up[layer].data[i] = nnet->matrix[layer][1][i][0];
        }
    } 
    weights_low[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_low_gtzero[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_low_gtzero[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_up[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_low[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_low[nnet->numLayers].row * weights_low[nnet->numLayers].col);
    memset(weights_low[nnet->numLayers].data, 0, sizeof(float)*weights_low[nnet->numLayers].row * weights_low[nnet->numLayers].col);
    weights_low_gtzero[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_low_gtzero[nnet->numLayers].row * weights_low_gtzero[nnet->numLayers].col);
    memset(weights_low_gtzero[nnet->numLayers].data, 0, sizeof(float)*weights_low_gtzero[nnet->numLayers].row * weights_low_gtzero[nnet->numLayers].col);
    weights_up[nnet->numLayers].data =\
        (float*)SAFEMALLOC(sizeof(float)*weights_up[nnet->numLayers].row * weights_up[nnet->numLayers].col);
    memset(weights_up[nnet->numLayers].data, 0, sizeof(float)*weights_up[nnet->numLayers].row * weights_up[nnet->numLayers].col);
    bias_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_low_gtzero[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_low[nnet->numLayers].row = (float)1;
    bias_low_gtzero[nnet->numLayers].row = (float)1;
    bias_up[nnet->numLayers].row = (float)1;
    bias_low[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_low[nnet->numLayers].col);
    memset(bias_low[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);
    bias_low_gtzero[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_low_gtzero[nnet->numLayers].col);
    memset(bias_low_gtzero[nnet->numLayers].data, 0, sizeof(float)*bias_low_gtzero[nnet->numLayers].col);
    bias_up[nnet->numLayers].data = (float*)SAFEMALLOC(sizeof(float)*bias_up[nnet->numLayers].col);
    memset(bias_up[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);


    nnet->weights_low = weights_low;
    nnet->weights_low_gtzero = weights_low_gtzero;
    nnet->weights_up = weights_up;
    nnet->bias_low = bias_low;
    nnet->bias_low_gtzero = bias_low_gtzero;
    nnet->bias_up = bias_up;
    
    nnet->cache_equation_low = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_equation_up = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_bias_low = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_bias_up = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_valid = false;

    
    nnet->cache_equation_low_gtzero = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_equation_up_gtzero = (float *)SAFEMALLOC(sizeof(float)*((nnet->inputSize))*(nnet->maxLayerSize));
    nnet->cache_bias_low_gtzero = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_bias_up_gtzero = (float *)SAFEMALLOC(sizeof(float)*(1)*(nnet->maxLayerSize));
    nnet->cache_valid_gtzero = false;

    nnet->is_duplicate = true;

    nnet->input_interval = orig_nnet->input_interval;

    return nnet;
}



struct NNet *reset_conv_network(struct NNet *nnet)
{
    // Dont't copy weights and biases from orig_net! Those may have been
    // modified by ReLU relax operations
    struct Matrix *weights_low = nnet->weights_low;
    struct Matrix *weights_up = nnet->weights_up;
    struct Matrix *bias_low = nnet->bias_low;
    struct Matrix *bias_up = nnet->bias_up;

    for(int layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==1) continue;
        weights_low[layer].row = nnet->layerSizes[layer];
        weights_up[layer].row = nnet->layerSizes[layer];
        weights_low[layer].col = nnet->layerSizes[layer+1];
        weights_up[layer].col = nnet->layerSizes[layer+1];
        
        int n=0;
        for(int i=0;i<weights_low[layer].col;i++){
            for(int j=0;j<weights_low[layer].row;j++){
                float w = nnet->matrix[layer][0][i][j];
                weights_low[layer].data[n] = w;
                weights_up[layer].data[n] = w;
                n++;
            }
        }
        bias_low[layer].col = nnet->layerSizes[layer+1];
        bias_up[layer].col = nnet->layerSizes[layer+1];
        bias_low[layer].row = (float)1;
        bias_up[layer].row = (float)1;
        
        for(int i=0;i<bias_low[layer].col;i++){
            bias_low[layer].data[i] = nnet->matrix[layer][1][i][0];
            bias_up[layer].data[i] = nnet->matrix[layer][1][i][0];
        }
    } 
    weights_low[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    weights_up[nnet->numLayers].row = nnet->layerSizes[nnet->numLayers];
    weights_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    memset(weights_low[nnet->numLayers].data, 0, sizeof(float)*weights_low[nnet->numLayers].row * weights_low[nnet->numLayers].col);
    memset(weights_up[nnet->numLayers].data, 0, sizeof(float)*weights_up[nnet->numLayers].row * weights_up[nnet->numLayers].col);
    bias_low[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_up[nnet->numLayers].col = nnet->layerSizes[nnet->numLayers + 1];
    bias_low[nnet->numLayers].row = (float)1;
    bias_up[nnet->numLayers].row = (float)1;
    memset(bias_low[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);
    memset(bias_up[nnet->numLayers].data, 0, sizeof(float)*bias_low[nnet->numLayers].col);


    nnet->cache_valid = false;
    nnet->cache_valid_gtzero = false;

    return nnet;
}

void destroy_conv_network(struct NNet *nnet)
{
    int i=0, row=0;
    if (nnet!=NULL)
    {
        for(i=0; i<nnet->numLayers; i++)
        {
            if(nnet->layerTypes[i]==1) continue;
            if(!nnet->is_duplicate) {
                for(row=0;row<nnet->layerSizes[i+1];row++)
                {
                    //free weight and bias arrays
                    free(nnet->matrix[i][0][row]);
                    free(nnet->matrix[i][1][row]);
                }
                //free pointer to weights and biases
                free(nnet->matrix[i][0]);
                free(nnet->matrix[i][1]);
                free(nnet->matrix[i]);
            }

            free(nnet->weights_low[i].data);
            free(nnet->weights_low_gtzero[i].data);
            free(nnet->weights_up[i].data);
            free(nnet->bias_low[i].data);
            free(nnet->bias_low_gtzero[i].data);
            free(nnet->bias_up[i].data);
        }
        free(nnet->weights_low[nnet->numLayers].data);
        free(nnet->weights_low_gtzero[nnet->numLayers].data);
        free(nnet->weights_up[nnet->numLayers].data);
        free(nnet->bias_low[nnet->numLayers].data);
        free(nnet->bias_low_gtzero[nnet->numLayers].data);
        free(nnet->bias_up[nnet->numLayers].data);

        if(!nnet->is_duplicate) {
            for(i=0;i<nnet->convLayersNum;i++){
                int in_channel = nnet->convLayer[i][1];
                int out_channel = nnet->convLayer[i][0];
                for(int oc=0;oc<out_channel;oc++){
                    for(int ic=0;ic<in_channel;ic++){
                        free(nnet->conv_matrix[i][oc][ic]);
                    }
                    free(nnet->conv_matrix[i][oc]);
                }
                free(nnet->conv_matrix[i]);
                free(nnet->conv_bias[i]);
            }
            free(nnet->conv_bias);
            free(nnet->conv_matrix);
            for(i=0;i<nnet->convLayersNum;i++){
                free(nnet->convLayer[i]);
            }
            free(nnet->convLayer);
            free(nnet->layerSizes);
            free(nnet->layerTypes);
            free(nnet->matrix);
        }
        free(nnet->weights_low);
        free(nnet->weights_low_gtzero);
        free(nnet->weights_up);
        free(nnet->bias_low);
        free(nnet->bias_low_gtzero);
        free(nnet->bias_up);
        free(nnet->cache_equation_low);
        free(nnet->cache_equation_up);
        free(nnet->cache_bias_low);
        free(nnet->cache_bias_up);
        free(nnet->cache_equation_low_gtzero);
        free(nnet->cache_equation_up_gtzero);
        free(nnet->cache_bias_low_gtzero);
        free(nnet->cache_bias_up_gtzero);
        free(nnet);
    }
}


void sort(float *array, int num, int *ind){
    float tmp;
    int tmp_ind;
    for(int i = 0; i < num; i++){
        //printf("%d, %d, %d\n", i, ind[i], array[ind[i]]);
        array[i] = array[ind[i]];
    }
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < (num - i - 1); j++)
        {
            if (array[j] < array[j + 1])
            {
                tmp = array[j];
                tmp_ind = ind[j];
                array[j] = array[j + 1];
                ind[j] = ind[j+1];
                array[j + 1] = tmp;
                ind[j+1] = tmp_ind;
            }
        }
    }
}


void sort_layers(int numLayers, int *layerSizes, int wrong_node_length, int *wrong_nodes_map){
	int wrong_nodes_tmp[wrong_node_length];
	memset(wrong_nodes_tmp, 0, sizeof(int)*wrong_node_length);
	int j = 0;
	int count_node = 0;
	for (int layer = 1; layer < numLayers; layer++){
		count_node += layerSizes[layer]; 
        // printf("%d, %d\n", count_node, count_node-layerSizes[layer]);
		for (int i = 0; i < wrong_node_length; i++){
			if(wrong_nodes_map[i]<count_node && wrong_nodes_map[i]>=count_node-layerSizes[layer]){
				wrong_nodes_tmp[j] = wrong_nodes_map[i];
                //printf("%d, %d;", j, wrong_nodes_tmp[j]);
				j++;
			}
		}
		
	}
    memcpy(wrong_nodes_map, wrong_nodes_tmp, sizeof(int)*wrong_node_length);
}


void set_input_constraints(struct Interval *input,
                        lprec *lp, int *rule_num, int inputSize)
{
    int Ncol = inputSize;
    REAL row[1];
    int colno[1];
    set_add_rowmode(lp, TRUE);
    for(int var=1;var<Ncol+1;var++){
        colno[0] = var;
        row[0] = 1;
        add_constraintex(lp, 1, row, colno, LE,\
                    input->upper_matrix.data[var-1]);
        add_constraintex(lp, 1, row, colno, GE,\
                    input->lower_matrix.data[var-1]);
        *rule_num += 2;
    }
    set_add_rowmode(lp, FALSE);
}


void set_node_constraints(lprec *lp, float *equation, float bias,
                        int start, int *rule_num,
                        int sig, int inputSize)
{
    int Ncol = inputSize;
    REAL row[Ncol+1];
    set_add_rowmode(lp, TRUE);
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start+j-1];
    }
    if(sig==1){
        add_constraintex(lp, 1, row, NULL, GE, -bias);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE, -bias);
    }
    *rule_num += 1;
    set_add_rowmode(lp, FALSE);
}


float set_output_constraints(lprec *lp, float *equation, float bias,
                int start_place, int *rule_num, int inputSize,
                int is_max, float *output, float *input_prev)
{
    int Ncol = inputSize;
    REAL row[Ncol+1];
    set_add_rowmode(lp, TRUE);
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start_place+j-1];
    }
    if(is_max){
        add_constraintex(lp, 1, row, NULL, GE,\
                    -bias);
        set_maxim(lp);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE,\
                    -bias);
        set_minim(lp);
    }
    *rule_num += 1;
    
    set_add_rowmode(lp, FALSE);
    
    set_obj_fnex(lp, Ncol, row, NULL);
    //write_lp(lp, "model3.lp");
    int ret = 0;

    //printf("in1\n");

    set_timeout(lp, 30);
    ret = solve(lp);

    //printf("in2,%d\n",ret);
    
    int feasible = 0;
    if(ret == OPTIMAL || ret == SUBOPTIMAL){
        int Ncol = inputSize;
        double row[Ncol+1];
        *output = get_objective(lp) + equation[inputSize+start_place];
        get_variables(lp, row);
        for(int j=0;j<Ncol;j++){
            input_prev[j] = (float)row[j];
        }
        feasible = 1;
    }
    else if(ret == TIMEOUT) {
        feasible = -1;
        printf("LP solving: timeout \n");
    }
    else if(ret == NUMFAILURE) {
        feasible = -1;
        printf("LP solving: numerical failure \n");
    }
    else if(ret == ACCURACYERROR) {
        feasible = -1;
        printf("LP solving: accuracy error \n");
    }
    else if(ret == UNBOUNDED) {
        feasible = -1;
        printf("LP solving: unbounded. LP printout follows: \n");
        print_lp(lp);
    }
    else if(ret == INFEASIBLE) {
        feasible = 0;
    }
    else {
        printf("Abort - Unexpected LP solver return value: %d \n", ret);
        exit(1);
    }
    
    del_constraint(lp, *rule_num);
    *rule_num -= 1; 
    
    return feasible;
}

float set_wrong_node_constraints(lprec *lp,
                float *equation, int start, int *rule_num,
                int inputSize, int is_max, float *output)
{

    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    row[0] = 0;
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start+j-1];
    }
    if(is_max){
        set_maxim(lp);
    }
    else{
        set_minim(lp);
    }
    
    set_obj_fnex(lp, Ncol+1, row, NULL);
    int ret = solve(lp);
    if(ret == OPTIMAL){
        int Ncol = inputSize;
        REAL row[Ncol+1];
        get_variables(lp, row);
        *output = get_objective(lp)+equation[inputSize+start];
    }
    else{
        //printf("unsat!\n");
        unsat = 1;
    }
    return unsat;
}


void initialize_input_interval(struct NNet* nnet,
                int img, int inputSize, float *input,
                float *u, float *l)
{
    load_inputs(img, inputSize, input);
    if(PROPERTY == 0){
        int demo = 0;
        if(demo == 1) {
            u[0] = 6;
            l[0] = 4;
            u[1] = 4.1;
            l[1] = 3;
            u[2] = 0;
            l[2] = 0;
        }
        if(demo == 2) {
            u[0] = 0;
            l[0] = 0;
            u[1] = 4;
            l[1] = -4;
        }
        for(int i =0;demo==0 && i<inputSize;i++){
            u[i] = input[i]+INF;
            if(u[i] > nnet->max) {
                u[i] = nnet->max;
            }
            l[i] = input[i]-INF;
            if(l[i] < nnet->min) {
                l[i] = nnet->min;
            }
        }
        // used for biases
        u[inputSize] = 1;
        l[inputSize] = 1;
    }
    else if(PROPERTY == 1){
        /*
         * Customize your own initial input range
         */
    }
    else{
        for(int i =0;i<inputSize;i++){
            u[i] = input[i]+INF;
            l[i] = input[i]-INF;
        }
    }

}


void load_inputs(int img, int inputSize, float *input){

    if(img>=100000){
        printf("ERR: Over 100000 images!\n");
        exit(1);
    }
    char str[12];
    char image_name[18];

    /*
     * Customize your own dataset
     */

    char tmp[18] = "images/image";
    strcpy(image_name, tmp);

    sprintf(str, "%d", img);
    FILE *fstream = fopen(strcat(image_name,str),"r");
    if (fstream == NULL)
    {
        printf("no input:%s!\n", image_name);
        exit(1);
    }
    int bufferSize = 10240*5;
    char *buffer = (char*)SAFEMALLOC(sizeof(char)*bufferSize);
    char *record, *line;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (int i = 0; i<inputSize; i++)
    {
        input[i] = atof(record);
        record = strtok(NULL,",\n");
    }
    free(buffer);
    fclose(fstream);

}


void denormalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {
        input->data[i] = input->data[i] * (nnet->max - nnet->min) + nnet->min;
    }

    /*
     * You might want to customize the denormalization function as well
     */
}


void denormalize_input_interval(struct NNet *nnet, struct Interval *input){
    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);
}


void normalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = (input->data[i] - nnet->min) / (nnet->max - nnet->min);
    }

    /*
     * You might want to customize the normalization function as well
     */
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input){
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}


void forward_prop_conv(struct NNet *network,
            struct Matrix *input, struct Matrix *output){
    evaluate_conv(network, input, output);
    float t = output->data[network->target];
    for(int o=0;o<network->outputSize;o++){
        output->data[o] -= t; 
    }
}



int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output){
    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;
    float ****conv_matrix = nnet->conv_matrix;

    float tempVal;
    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];

    
    //printf("start evaluate\n");
    for (i=0; i < nnet->inputSize; i++)
    {
        z[i] = input->data[i];
    }

    int out_channel=0, in_channel=0, kernel_size=0;
    int stride=0, padding=0;

    for (layer = 0; layer<(numLayers); layer++)
    {

        memset(a, 0, sizeof(float)*nnet->maxLayerSize);

        //printf("layer:%d %d\n",layer, nnet->layerTypes[layer]);        
        if(nnet->layerTypes[layer]==0){
            for (i=0; i < nnet->layerSizes[layer+1]; i++){
                float **weights = matrix[layer][0];
                float **biases  = matrix[layer][1];
                tempVal = 0.0;

                //Perform weighted summation of inputs
                for (j=0; j<nnet->layerSizes[layer]; j++){
                    tempVal += z[j]*weights[i][j];
                }

                //Add bias to weighted sum
                tempVal += biases[i][0];

                //Perform ReLU
                if (tempVal<0.0 && layer<(numLayers-1)){
                    // printf( "doing RELU on layer %u\n", layer );
                    tempVal = 0.0;
                }
                a[i]=tempVal;
            }
            for(j=0;j<nnet->maxLayerSize;j++){
                //if(layer==2 && j<100) printf("%d %f\n",j, a[j]);
                //if(layer==5 && j<100) printf("%d %f\n",j, a[j]);
                z[j] = a[j];
            }
        }
        else{
            out_channel = nnet->convLayer[layer][0];
            in_channel = nnet->convLayer[layer][1];
            kernel_size = nnet->convLayer[layer][2];
            stride = nnet->convLayer[layer][3];
            padding = nnet->convLayer[layer][4];
            //size is the input size in each channel
            int size = sqrt(nnet->layerSizes[layer]/in_channel);
            //padding size is the input size after padding
            int padding_size = size+2*padding;
            //this is only for compressed model
            if(kernel_size%2==1){
                padding_size += 1;
            }
            //out_size is the output size in each channel after kernel
            int out_size = 0;

            /*
             * If you find your padding strategies are different from the one implemented.
             * You might want to change it here.
             */
            float tmp_out_size =  (padding_size-(kernel_size-1)-1)/stride+1;
            if(tmp_out_size == (int)tmp_out_size){
                out_size = (int)tmp_out_size;
            }
            else{
                out_size = (int)(tmp_out_size)-1;
            }

            float *z_new = (float*)SAFEMALLOC(sizeof(float)*padding_size*padding_size*in_channel);
            memset(z_new, 0, sizeof(float)*padding_size*padding_size*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                for(int h=0;h<size;h++){
                    for(int w=0;w<size;w++){
                        z_new[ic*padding_size*padding_size+padding_size*(h+padding)+w+padding] =\
                                                            z[ic*size*size+size*h+w];
                    }
                }
            }

            for(int oc=0;oc<out_channel;oc++){
                for(int oh=0;oh<out_size;oh++){
                    for(int ow=0;ow<out_size;ow++){
                        int start = ow*stride+oh*stride*padding_size;
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){
                                    a[oc*out_size*out_size+oh*out_size+ow] +=\
                                    conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                    z_new[ic*padding_size*padding_size+padding_size*kh+kw+start];
                                }
                            }
                        }
                        a[oc*out_size*out_size+ow+oh*out_size]+=nnet->conv_bias[layer][oc];
                    }
                }
            }
            for(j=0;j<nnet->maxLayerSize;j++){
                
                if(a[j]<0){
                    a[j] = 0;
                }
                z[j] = a[j];

            }
            free(z_new);
        }
    }

    for (i=0; i<outputSize; i++){
        output->data[i] = a[i];
    }
    
    return 1;
}


void backward_prop_conv(struct NNet *nnet, float *grad,
                     int R[][nnet->maxLayerSize]){
    int i, j, layer;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int maxLayerSize   = nnet->maxLayerSize;

    float grad_upper[maxLayerSize];
    float grad_lower[maxLayerSize];
    float grad1_upper[maxLayerSize];
    float grad1_lower[maxLayerSize];
    memcpy(grad_upper, nnet->matrix[numLayers-1][0][nnet->target],\
             sizeof(float)*nnet->layerSizes[numLayers-1]);
    memcpy(grad_lower, nnet->matrix[numLayers-1][0][nnet->target],\
             sizeof(float)*nnet->layerSizes[numLayers-1]);
    int start_node = 0;
    for(int l=1; l<nnet->numLayers-1;l++){
        start_node += nnet->layerSizes[l];
    }
    for(int i=0;i<nnet->layerSizes[nnet->numLayers-1];i++){
        //printf("%d, %f %f\n", start_node+i, grad_upper[i], grad_lower[i]);
        grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
        //printf("%d, %f\n", start_node+i, grad[start_node+i]);
    }

    for(layer = numLayers-2;layer>-1;layer--){
        //printf("layer:%d , %d\n", layer, nnet->layerTypes[layer]);
        float **weights = nnet->matrix[layer][0];
        memset(grad1_upper, 0, sizeof(float)*maxLayerSize);
        memset(grad1_lower, 0, sizeof(float)*maxLayerSize);

        if(nnet->layerTypes[layer]!=1){
            if(layer != 0){
                for(j=0;j<nnet->layerSizes[numLayers-1];j++){
                    if(R[layer][j]==0){
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if(R[layer][j]==1){
                        grad_upper[j] = (grad_upper[j]>0)?grad_upper[j]:0;
                        grad_lower[j] = (grad_lower[j]<0)?grad_lower[j]:0;
                    }

                    for(i=0;i<nnet->layerSizes[numLayers-1];i++){
                        if(weights[j][i]>=0){
                            grad1_upper[i] += weights[j][i]*grad_upper[j]; 
                            grad1_lower[i] += weights[j][i]*grad_lower[j]; 
                        }
                        else{
                            grad1_upper[i] += weights[j][i]*grad_lower[j]; 
                            grad1_lower[i] += weights[j][i]*grad_upper[j]; 
                        }
                    }
                }
            }
            else{
                for(j=0;j<nnet->layerSizes[numLayers-1];j++){
                    if(R[layer][j]==0){
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if(R[layer][j]==1){
                        grad_upper[j] = (grad_upper[j]>0)?grad_upper[j]:0;
                        grad_lower[j] = (grad_lower[j]<0)?grad_lower[j]:0;
                    }

                    for(i=0;i<inputSize;i++){
                        if(weights[j][i]>=0){
                            grad1_upper[i] += weights[j][i]*grad_upper[j]; 
                            grad1_lower[i] += weights[j][i]*grad_lower[j]; 
                        }
                        else{
                            grad1_upper[i] += weights[j][i]*grad_lower[j]; 
                            grad1_lower[i] += weights[j][i]*grad_upper[j]; 
                        }
                    }
                }
            }
        }
        else{
            break;
        }

        
        if(layer!=0 && nnet->layerTypes[layer-1]!=1){
            //printf("%d, %d\n", layer, nnet->layerSizes[layer]);
            memcpy(grad_upper,grad1_upper,sizeof(float)*nnet->layerSizes[numLayers-1]);
            memcpy(grad_lower,grad1_lower,sizeof(float)*nnet->layerSizes[numLayers-1]);
            int start_node = 0;
            for(int l=1; l<layer;l++){
                start_node += nnet->layerSizes[l];
            }
            for(int i=0;i<nnet->layerSizes[layer];i++){
                grad[start_node+i] = (grad_upper[i]>-grad_lower[i])?grad_upper[i]:-grad_lower[i];
                //printf("%d, %f %f\n", start_node+i, grad_upper[i], grad_lower[i]);
                //printf("%d, %f\n", start_node+i, grad[start_node+i]);
            }
        }
        else{
            break;
            //memcpy(grad->lower_matrix.data, grad1_lower, sizeof(float)*inputSize);
            //memcpy(grad->upper_matrix.data, grad1_upper, sizeof(float)*inputSize);
        }
    }
}


void update_equations(struct NNet *nnet, int layer, bool use_gtzero_lb) {
    int inputSize    = nnet->inputSize;
    int maxLayerSize   = nnet->maxLayerSize;


    float *cache_equation_low;
    float *cache_equation_up;
    float *cache_bias_low;
    float *cache_bias_up;

    if(use_gtzero_lb) {
        if(nnet->cache_valid_gtzero) {
            return;
        }
        cache_equation_low = nnet->cache_equation_low_gtzero;
        cache_equation_up = nnet->cache_equation_up_gtzero;
        cache_bias_low = nnet->cache_bias_low_gtzero;
        cache_bias_up = nnet->cache_bias_up_gtzero;
    }
    else {
        if(nnet->cache_valid) {
            return;
        }
        cache_equation_low = nnet->cache_equation_low;
        cache_equation_up = nnet->cache_equation_up;
        cache_bias_low = nnet->cache_bias_low;
        cache_bias_up = nnet->cache_bias_up;
    }

    memset(cache_equation_low, 0, sizeof(float)*(inputSize)*maxLayerSize);
    memset(cache_equation_up, 0, sizeof(float)*(inputSize)*maxLayerSize);
    memset(cache_bias_low, 0, sizeof(float)*(1)*maxLayerSize);
    memset(cache_bias_up, 0, sizeof(float)*(1)*maxLayerSize);

    if(nnet->layerSizes[0] != inputSize) {
        printf("Error B: %d != %d \n", nnet->layerSizes[layer], inputSize);
        exit(1);
    }


    memcpy(cache_equation_low, 
        nnet->weights_low[layer].data,
        sizeof(float) * nnet->layerSizes[layer + 1] * nnet->layerSizes[layer]);
    memcpy(cache_equation_up, 
        nnet->weights_up[layer].data,
        sizeof(float) * nnet->layerSizes[layer + 1] * nnet->layerSizes[layer]);
    memcpy(cache_bias_low, nnet->bias_low[layer].data, sizeof(float) * nnet->layerSizes[layer + 1]);
    memcpy(cache_bias_up, nnet->bias_up[layer].data, sizeof(float) * nnet->layerSizes[layer + 1]);

    float *equation_low_pos = (float*)SAFEMALLOC(sizeof(float) *\
                            (inputSize)*maxLayerSize);
    float *equation_low_neg = (float*)SAFEMALLOC(sizeof(float) *\
                            (inputSize)*maxLayerSize);
    float *equation_up_pos = (float*)SAFEMALLOC(sizeof(float) *\
                            (inputSize)*maxLayerSize);
    float *equation_up_neg = (float*)SAFEMALLOC(sizeof(float) *\
                            (inputSize)*maxLayerSize);

    for(int l=layer; l>0; l--) {
        memset(equation_low_pos, 0, sizeof(float)*(inputSize)*maxLayerSize);
        memset(equation_low_neg, 0, sizeof(float)*(inputSize)*maxLayerSize);
        memset(equation_up_pos, 0, sizeof(float)*(inputSize)*maxLayerSize);
        memset(equation_up_neg, 0, sizeof(float)*(inputSize)*maxLayerSize);

        //printf("Intermediate upper equations: \n");
        //for(int j = 0; j<maxLayerSize; j++) {
        //    printf("%d: ", j);
        //    for(int i = 0; i < inputSize+1; i++) {
        //        printf("%f ", equation_up[i + j * (inputSize+1)]);
        //    }
        //    printf("\n");
        //}
//
        //printf("Intermediate lower equations: \n");
        //for(int j = 0; j<maxLayerSize; j++) {
        //    printf("%d: ", j);
        //    for(int i = 0; i < inputSize+1; i++) {
        //        printf("%f ", equation_low[i + j * (inputSize+1)]);
        //    }
        //    printf("\n");
        //}
        //printf("\n");



        int n = 0;
        for(int j = 0; j < nnet->layerSizes[layer+1]; j++) {
            for(int i = 0; i < nnet->layerSizes[l]; i++) {
                if(cache_equation_low[n] > 0) {
                    equation_low_pos[n] = cache_equation_low[n];
                }
                else {
                    equation_low_neg[n] = cache_equation_low[n];
                }

                if(cache_equation_up[n] > 0) {
                    equation_up_pos[n] = cache_equation_up[n];
                }
                else {
                    equation_up_neg[n] = cache_equation_up[n];
                }

                n++;
            }
        }

        memset(cache_equation_low, 0, sizeof(float)*(inputSize)*maxLayerSize);
        memset(cache_equation_up, 0, sizeof(float)*(inputSize)*maxLayerSize);

        struct Matrix equation_low_pos_matrix = {
            equation_low_pos, nnet->layerSizes[layer+1], nnet->layerSizes[l]
        };
        struct Matrix equation_low_neg_matrix = {
            equation_low_neg, nnet->layerSizes[layer+1], nnet->layerSizes[l]
        };
        struct Matrix equation_up_pos_matrix = {
            equation_up_pos, nnet->layerSizes[layer+1], nnet->layerSizes[l]
        };
        struct Matrix equation_up_neg_matrix = {
            equation_up_neg, nnet->layerSizes[layer+1], nnet->layerSizes[l]
        };


        struct Matrix equation_low_matrix = {
            cache_equation_low, nnet->layerSizes[layer+1], nnet->layerSizes[l-1]
        };
        struct Matrix equation_up_matrix = {
            cache_equation_up, nnet->layerSizes[layer+1], nnet->layerSizes[l-1]
        };

        struct Matrix lower_weights = nnet->weights_low[l-1];
        struct Matrix lower_bias = nnet->bias_low[l-1];
        if(use_gtzero_lb && l == layer) {
            //printf("Using gt weights \n");
            lower_weights = nnet->weights_low_gtzero[l-1];
            lower_bias = nnet->bias_low_gtzero[l-1];
        }
        //if(!use_gtzero_lb && l == layer) {
        //    lower_weights = nnet->weights_low_gtzero[l-1];
        //    lower_bias = nnet->bias_low_gtzero[l-1];
        //}

        matmul(&lower_weights, &equation_low_pos_matrix, &equation_low_matrix);
        matmul_with_bias(&(nnet->weights_up[l-1]), &equation_low_neg_matrix, &equation_low_matrix);

        matmul(&(nnet->weights_up[l-1]), &equation_up_pos_matrix, &equation_up_matrix);
        matmul_with_bias(&lower_weights, &equation_up_neg_matrix, &equation_up_matrix);

        

        struct Matrix bias_low_matrix = {
            cache_bias_low, nnet->layerSizes[layer+1], 1
        };
        struct Matrix bias_up_matrix = {
            cache_bias_up, nnet->layerSizes[layer+1], 1
        };


        matmul_with_bias(&lower_bias, &equation_low_pos_matrix, &bias_low_matrix);
        matmul_with_bias(&(nnet->bias_up[l-1]), &equation_low_neg_matrix, &bias_low_matrix);

        matmul_with_bias(&(nnet->bias_up[l-1]), &equation_up_pos_matrix, &bias_up_matrix);
        matmul_with_bias(&lower_bias, &equation_up_neg_matrix, &bias_up_matrix);
    }

    free(equation_low_pos);
    free(equation_low_neg);
    free(equation_up_pos);
    free(equation_up_neg);

    if(use_gtzero_lb) {
        nnet->cache_valid_gtzero = true;
    }
    else {
        nnet->cache_valid = true;
    }
    

    //printf("Final returned upper equations: \n");
    //for(int j = 0; j<maxLayerSize; j++) {
    //    printf("%d: ", j);
    //    for(int i = 0; i < inputSize+1; i++) {
    //        printf("%f ", equation_up[i + j * (inputSize+1)]);
    //    }
    //    printf("\n");
    //}
//
    //printf("Final returned lower equations: \n");
    //for(int j = 0; j<maxLayerSize; j++) {
    //    printf("%d: ", j);
    //    for(int i = 0; i < inputSize+1; i++) {
    //        printf("%f ", equation_low[i + j * (inputSize+1)]);
    //    }
    //    printf("\n");
    //}
    //printf("\n");

}


void sym_fc_layer(struct NNet *nnet,
                    int layer, int err_row) {
    printf("No longer needed. Remove! \n");
    exit(1);
}


void sym_conv_layer(struct SymInterval *sInterval,
                    struct SymInterval *new_sInterval,
                    struct NNet *nnet, int layer, int err_row) {
    printf("Not implemented! \n");
    exit(1);
}


// calculate the upper and lower bound for the ith node in each layer
void relu_bound(struct NNet *nnet, 
                struct Interval *input, int i, int layer, int err_row, 
                float *low, float *up, int ignore, bool use_gtzero_lb){
    int inputSize    = nnet->inputSize;
    
    update_equations(nnet, layer, use_gtzero_lb);

    
    //ignore = 0;
    float tempVal_upper=0.0, tempVal_lower=0.0;
    
    float needed_outward_round = 0;
    if(NEED_OUTWARD_ROUND) {
        needed_outward_round = OUTWARD_ROUND;
    }

    float *cache_equation_low;
    float *cache_equation_up;
    float *cache_bias_low;
    float *cache_bias_up;

    if(use_gtzero_lb) {
        if(!nnet->cache_valid_gtzero) {
            printf("Cached equations not up to date? \n");
            exit(1);
        }
        cache_equation_low = nnet->cache_equation_low_gtzero;
        cache_equation_up = nnet->cache_equation_up_gtzero;
        cache_bias_low = nnet->cache_bias_low_gtzero;
        cache_bias_up = nnet->cache_bias_up_gtzero;
    }
    else {
        if(!nnet->cache_valid) {
            printf("Cached equations not up to date? \n");
            exit(1);
        }
        cache_equation_low = nnet->cache_equation_low;
        cache_equation_up = nnet->cache_equation_up;
        cache_bias_low = nnet->cache_bias_low;
        cache_bias_up = nnet->cache_bias_up;
    }

    
    for(int k=0;k<inputSize;k++){
        float weight_low = cache_equation_low[k+i*(inputSize)];
        float weight_up = cache_equation_up[k+i*(inputSize)];

        if(ignore == 1) {
            weight_up = weight_low;
        }
        else if(ignore == 2) {
            weight_low = weight_up;
        }


        if(ignore == 1) {
            weight_up = weight_low;
        }
        else if(ignore == 2) {
            weight_low = weight_up;
        }

        if(weight_low>=0){
            tempVal_lower +=\
                weight_low * input->lower_matrix.data[k]-needed_outward_round;
        }
        else{
            tempVal_lower +=\
                weight_low * input->upper_matrix.data[k]-needed_outward_round;
        } 

        if(weight_up>=0){
            tempVal_upper +=\
                weight_up * input->upper_matrix.data[k]+needed_outward_round;
        }
        else{
            tempVal_upper +=\
                weight_up * input->lower_matrix.data[k]+needed_outward_round;
        } 
    }

    if(ignore == 0) {
        tempVal_lower += cache_bias_low[i];
        tempVal_upper += cache_bias_up[i];
    }
    else if(ignore == 1) {
        tempVal_lower += cache_bias_low[i];
        tempVal_upper += cache_bias_low[i];
    }
    else if(ignore == 2) {
        tempVal_lower += cache_bias_up[i];
        tempVal_upper += cache_bias_up[i];
    }
    else {
        printf("Invalid ignore parameter \n");
        exit(1);
    }

    *up = tempVal_upper;
    *low = tempVal_lower;

    //printf("Bounds layer %d/%d: %f - %f \n", layer, i, *low, *up);
}

int relax_relu(struct NNet *nnet, 
    float low_lower_bound, float low_upper_bound,
    float up_lower_bound, float up_upper_bound, float low_temp_lower_gt,
    struct Interval *input, int i, int layer,
    int *err_row, int *wrong_node_length, int *wcnt, bool ignore_invalid_output) {

    if(low_temp_lower_gt < low_lower_bound) {
        low_temp_lower_gt = low_lower_bound;
    }
    

    if(low_lower_bound > low_upper_bound || up_lower_bound > up_upper_bound) {
        printf("Invalid bounds \n");
        exit(1);
    }

    if(low_lower_bound > up_lower_bound || low_upper_bound > up_upper_bound) {
        printf("Invalid (switched) bounds \n");
        printf("(%f - %f) - (%f - %f) \n", low_lower_bound, low_upper_bound, up_lower_bound, up_upper_bound);
        //printf("Upper: "); printMatrix(sym_interval->matrix_up);
        //printf("Lower: "); printMatrix(sym_interval->matrix_low);
        //printf("Error: "); printMatrix(sym_interval->err_matrix);
        exit(1);
    }
    //else {
    //    printf("Alright bounds \n");
    //    printf("(%f - %f) - (%f - %f) \n", low_lower_bound, low_upper_bound, up_lower_bound, up_upper_bound);
    //}

    //printf("relu relaxation\n");
    int result = -1;
    if (up_upper_bound<=0.0){
        if(nnet->weights_low[layer].row != nnet->layerSizes[layer]) {
            printf("Unexpted sizes A \n");
            exit(1);
        }
        for(int k=0; k < nnet->weights_low[layer].row; k++){
            nnet->weights_low[layer].data[k + i*nnet->layerSizes[layer]] = 0;
            nnet->weights_low_gtzero[layer].data[k + i*nnet->layerSizes[layer]] = 0;
            nnet->weights_up[layer].data[k + i*nnet->layerSizes[layer]] = 0;
        }
        nnet->bias_low[layer].data[i] = 0;
        nnet->bias_low_gtzero[layer].data[i] = 0;
        nnet->bias_up[layer].data[i] = 0;
        
        result = 0;
    }
    else if(low_lower_bound>=0.0){
        result = 2;
    }
    else{
        int actions = 0;

        if((low_lower_bound < 0 && low_upper_bound > 0)) {
            actions++;
            
            *wrong_node_length += 1;
            *wcnt += 1;
            *err_row += 1;
        }
        if((low_temp_lower_gt < 0 && up_upper_bound > 0)) {
            actions++;
            
            *wrong_node_length += 1;
            *wcnt += 1;
            *err_row += 1;

            float scaling = up_upper_bound / (up_upper_bound - low_temp_lower_gt);
            for(int k=0; k < nnet->weights_up[layer].row; k++){
                nnet->weights_up[layer].data[k + i*nnet->layerSizes[layer]] *= scaling;
            }
            nnet->bias_up[layer].data[i] *= scaling;
            

            nnet->bias_up[layer].data[i] -= \
                low_temp_lower_gt*scaling;
        }

        if(actions == 2) {
            *wcnt -= 1;
            *wrong_node_length -= 1;
        }

        if(low_upper_bound <= -1.4 * low_lower_bound) {
            for(int k=0; k < nnet->weights_low[layer].row; k++){
                nnet->weights_low_gtzero[layer].data[k + i*nnet->layerSizes[layer]] = 0;
            }
            nnet->bias_low_gtzero[layer].data[i] = 0;
        }  
//

        if(low_upper_bound <= 0) {
            for(int k=0; k < nnet->weights_low[layer].row; k++){
                nnet->weights_low[layer].data[k + i*nnet->layerSizes[layer]] = 0;
                nnet->weights_low_gtzero[layer].data[k + i*nnet->layerSizes[layer]] = 0;
            }
            nnet->bias_low[layer].data[i] = 0;
            nnet->bias_low_gtzero[layer].data[i] = 0;

            actions++;
            low_upper_bound=99999999; // prevent double execution (see below)
        }  
//
        if(low_upper_bound <= -low_lower_bound) {
            for(int k=0; k < nnet->weights_low[layer].row; k++){
                nnet->weights_low[layer].data[k + i*nnet->layerSizes[layer]] = 0;
                nnet->weights_low_gtzero[layer].data[k + i*nnet->layerSizes[layer]] = 0;
            }
            nnet->bias_low[layer].data[i] = 0;
            nnet->bias_low_gtzero[layer].data[i] = 0;

            actions++;
            low_upper_bound=99999999; // prevent double execution (see below)
        }  

        if(actions == 0) {
            printf("Why was no action applied? %f - %f, %f - %f \n",
            low_lower_bound, low_upper_bound, up_lower_bound, up_upper_bound);
            exit(1);
        }


        
        result = 10 + actions;
    }

    //float low_tempVal_upper = 0;
    //float low_tempVal_lower = 0;
    //float up_tempVal_upper = 0;
    //float up_tempVal_lower = 0;

    //relu_bound(nnet, input, i, layer, *err_row, &up_tempVal_lower, &up_tempVal_upper, 2);
    //relu_bound(nnet, input, i, layer, *err_row, &low_tempVal_lower, &low_tempVal_upper, 1);


    //if(!ignore_invalid_output && up_tempVal_lower < 0) {
    //    if(up_tempVal_lower < 0) {
    //        printf("Lower bound of upper bound must never be less "
    //            "than zero, but is %.20f \n", up_tempVal_lower);
    //        exit(1);
    //    }
    //}

    if(result < 0) {
        printf("No relaxation action was taken?! \n");
    }
    return result;
}


// relax the relu layers and get the new symbolic equations
int sym_relu_layer(struct Interval *input,
                    struct Interval *output,
                    struct NNet *nnet, 
                    int R[][nnet->maxLayerSize],
                    int layer, int *err_row,
                    int *wrong_nodes_map, 
                    int *wrong_node_length,
                    int *node_cnt)
{
    //record the number of wrong nodes
    int wcnt = 0;

    for (int i=0; i < nnet->layerSizes[layer+1]; i++)
    {
        float tempVal_upper=0.0, tempVal_lower=0.0;
        relu_bound(nnet, input, i, layer, *err_row,\
                    &tempVal_lower, &tempVal_upper, 0, false);

        //printf("Layer %d, node %d: %f - %f \n", layer, i, tempVal_lower, tempVal_upper);
        
        //Perform ReLU relaxation
        if(layer == nnet->numLayers - 1) {
            output->upper_matrix.data[i] = tempVal_upper;
            output->lower_matrix.data[i] = tempVal_lower;
        }
        else {
            float low_tempVal_upper = 0;
            float low_tempVal_lower = 0;
            float up_tempVal_upper = 0;
            float up_tempVal_lower = 0;

            relu_bound(nnet, input, i, layer, *err_row, &up_tempVal_lower, &up_tempVal_upper, 2, false);
            relu_bound(nnet, input, i, layer, *err_row, &low_tempVal_lower, &low_tempVal_upper, 1, false);

            float low_temp_upper_gt = 0;
            float low_temp_lower_gt = 0;
            relu_bound(nnet, input, i, layer, *err_row, &low_temp_lower_gt, &low_temp_upper_gt, 1, true);

            if(low_tempVal_lower < low_temp_lower_gt) {
                //printf("Saved: %f vs. %f \n", low_tempVal_lower, low_temp_lower_gt);
            }
            else {
                //printf("Didn't save: %f vs. %f \n", low_tempVal_lower, low_temp_lower_gt);
            }

            //printf("Before relax UP (%f - %f), LOW (%f - %f) \n",
            //    up_tempVal_lower, up_tempVal_upper, low_tempVal_lower, low_tempVal_upper);

            R[layer][i] = relax_relu(nnet, low_tempVal_lower, low_tempVal_upper, up_tempVal_lower, up_tempVal_upper,
                low_temp_lower_gt, input, i, layer, err_row, wrong_node_length, &wcnt, false);

            if(R[layer][i] > 10) {
                wrong_nodes_map[(*wrong_node_length) - 1] = *node_cnt;
                R[layer][i] = 1;
            }

            tempVal_upper=0.0, tempVal_lower=0.0;
            //relu_bound(nnet, input, i, layer, *err_row,
            //            &tempVal_lower, &tempVal_upper, 0);

            //printf("After ReLu: Layer %d, node %d: %f - %f \n", layer, i, tempVal_lower, tempVal_upper);


            
        }
        (*node_cnt) += 1;  
    }

    nnet->cache_valid = false;
    nnet->cache_valid_gtzero = false;

    return wcnt;
}


void forward_prop_interval_equation_linear_conv(struct NNet *nnet,
                            struct Interval *input,
                             struct Interval *output, float *grad,
                             int *wrong_nodes_map, int *wrong_node_length,
                             int *full_wrong_node_length)
{
    int node_cnt=0;

    int numLayers    = nnet->numLayers;
    int maxLayerSize   = nnet->maxLayerSize;
    
    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(int)*numLayers*maxLayerSize);

    //err_row is the number that is wrong before current layer
    int err_row=0;
    for (int layer = 0; layer<numLayers; layer++)
    {
        //printf("\n\nLayer %d \n", layer);
        //printf("Init Upper: "); printMatrix(sInterval.matrix_up);
        //printf("Init Lower: "); printMatrix(sInterval.matrix_low);
        //printf("Init Error: "); printMatrix(sInterval.err_matrix);


        //printf("Layer %d, %d", layer, nnet->layerSizes[layer]);

        if(nnet->layerTypes[layer]==0) {
            // FC layer
            
            //sym_fc_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
        
            //printf("new_sInterval Upper: "); printMatrix(new_sInterval.matrix_up);
            //printf("new_sInterval Lower: "); printMatrix(new_sInterval.matrix_low);
            //printf("new_sInterval Error: "); printMatrix(new_sInterval.err_matrix);
            
            
            int wcnt = sym_relu_layer(input, output, nnet, R,
                                layer, &err_row, wrong_nodes_map,
                                wrong_node_length, &node_cnt);
            
            *full_wrong_node_length = *full_wrong_node_length + wcnt;

        }
        else{

            //sym_conv_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
//
            //if(layer == 0){
//
            //    memcpy(equation_conv_low, new_equation_low,
            //            sizeof(float)*(inputSize+1)*maxLayerSize);
            //    memcpy(equation_conv_up, new_equation_up,
            //            sizeof(float)*(inputSize+1)*maxLayerSize);
            //    *err_row_conv = err_row;
//
            //}
//
            //sym_relu_layer(&new_sInterval, input, output, nnet, R, layer,
            //    &err_row, wrong_nodes_map, wrong_node_length, &node_cnt);

        }

        //printf("Layer equations: \n");
        //printf("Low: "); printMatrix(&new_equation_matrix_low);
        //printf("Up: "); printMatrix(&new_equation_matrix_up);
        //printf("Error: "); printMatrix(&new_equation_err_matrix);
        
        if(err_row >= ERR_NODE) {
            printf("err_row = %d > %d \n", err_row, ERR_NODE);
            exit(1);
        }
    }

    backward_prop_conv(nnet, grad, R);

}
