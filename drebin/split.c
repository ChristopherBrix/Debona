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

#include "split.h"

#define AVG_WINDOW 5
#define MAX_THREAD 56
#define MIN_DEPTH_PER_THREAD 5 

int NEED_PRINT = 0;
int NEED_FOR_ONE_RUN = 0;
int input_depth = 0;
int adv_found = 0;
int can_t_prove = 0;
int count = 0;
int thread_tot_cnt  = 0;
int smear_cnt = 0;

int progress = 0;

int CHECK_ADV_MODE = 0;
int PARTIAL_MODE = 0;

float avg_depth = 50;
float total_avg_depth = 0;
int leaf_num = 0;
float max_depth = 0;
int progress_list[PROGRESS_DEPTH];
int total_progress[PROGRESS_DEPTH];

int check_not_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_max_constant(struct NNet *nnet, struct Interval *output){
    if(output->upper_matrix.data[nnet->target]>0.5011){
        return 1;
    }
    else{
        return 0;
    }
}

int check_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->lower_matrix.data[i]>0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]<0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min_p7(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 3 && i != 4){
            if(output->upper_matrix.data[i]< output->lower_matrix.data[4] && output->upper_matrix.data[i]< output->lower_matrix.data[3])
                return 0;
        }
    }
    return 1;
}


int check_not_min_p8(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i!=0 && i!=1){
            if(output->lower_matrix.data[i]< output->upper_matrix.data[0] && output->lower_matrix.data[i]< output->upper_matrix.data[1]){
                return 1;
            }
        }
    }
    return 0;
}


int check_not_min(struct NNet *nnet, struct Interval *output){
	for(int i=0;i<nnet->outputSize;i++){
		if(output->lower_matrix.data[i]<0 && i != nnet->target){
			return 1;
		}
	}
	return 0;
}


int check_not_min_p11(struct NNet *nnet, struct Interval *output){

    if(output->lower_matrix.data[0]<0)
        return 1;

    return 0;
}


int check_min1_p7(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 3 && i != 4){
            if(output->data[i]<output->data[3] && output->data[i]<output->data[4])
                return 0;
        }
    }
    return 1;
}


int check_max_constant1(struct NNet *nnet, struct Matrix *output){
    if(output->data[nnet->target]<0.5011){
        return 0;
    }
    return 1;
}


int check_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return 0;
        }
    }
    return 1;
}


int check_not_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_not_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}


int check_not_min1_p8(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(i != 0 && i!=1){
            if(output->data[i]<output->data[0] && output->data[i]<output->data[1])
                return 1;
        }
    }
    return 0;
}


int check_not_min1_p11(struct NNet *nnet, struct Matrix *output){

    if(output->data[0]<0){
        return 1;
    }
    return 0;
}


int check_functions(struct NNet *nnet, struct Interval *output){
    return check_not_max(nnet, output);
}


int check_functions1(struct NNet *nnet, struct Matrix *output){
    return check_not_max1(nnet, output);
}


int tighten_still_overlap(struct NNet *nnet, struct Interval *input, float smear_sum){
    float out1[nnet->outputSize];
    struct Matrix output1 = {out1, nnet->outputSize, 1};
    float out2[nnet->outputSize];
    struct Matrix output2 = {out2, nnet->outputSize, 1};
    forward_prop(nnet, &input->lower_matrix, &output1);
    forward_prop(nnet, &input->upper_matrix, &output2);
    struct Interval output_interval ={output1, output2};
    float upper = 0;
    float lower = 0;
    for(int i=0;i<nnet->outputSize;i++){
        if(i!=nnet->target){
            if(output1.data[i]>output2.data[i]){
                lower = output1.data[i]-smear_sum;
                upper = output2.data[i]+smear_sum;
            }
            else{
                lower = output2.data[i]-smear_sum;
                upper = output1.data[i]+smear_sum;
            }
            
            output1.data[i] = lower;
            output2.data[i] = upper;
        }
    }
    return check_functions(nnet, &output_interval);
}


void *direct_run_check_thread(void *args){
    struct direct_run_check_args *actual_args = args;
    direct_run_check(actual_args->nnet, actual_args->input,
                     actual_args->output, actual_args->grad,
                     actual_args->depth, actual_args->feature_range,
                     actual_args->feature_range_length,
                     actual_args->split_feature,
                     actual_args->equation_upper,
                     actual_args->equation_lower,
                     actual_args->new_equation_upper,
                     actual_args->new_equation_lower);
    return NULL;
}


void *direct_run_check_lp_thread(void *args){
    struct direct_run_check_lp_args *actual_args = args;
    direct_run_check_lp(actual_args->nnet, actual_args->input,\
                     actual_args->grad, actual_args->output_map,
                     actual_args->equation_upper,\
                     actual_args->equation_lower,\
                     actual_args->new_equation_upper,\
                     actual_args->new_equation_lower,\
                     actual_args->wrong_nodes, actual_args->wrong_node_length,\
                     actual_args->sigs,\
                     actual_args->wrong_up_s_up,actual_args->wrong_up_s_low,\
                     actual_args->wrong_low_s_up, actual_args->wrong_low_s_low,\
                     actual_args->target, actual_args->sig,\
                     actual_args->lp, actual_args->rule_num, actual_args->depth);
    return NULL;
}

int direct_run_check(struct NNet *nnet, struct Interval *input, struct Interval *output,
                     struct Interval *grad, int depth, int *feature_range, 
                     int feature_range_length, int split_feature,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    forward_prop_interval_equation(nnet, input, output, grad,\
                                equation_upper, equation_lower,\
                                new_equation_upper, new_equation_lower);
    //forward_prop_interval_equation_linear2(nnet, input, output, grad,\
                                equation_upper, equation_lower,\
                                new_equation_upper, new_equation_lower);
    int isOverlap = check_functions(nnet, output);

    if(NEED_PRINT){
        pthread_mutex_lock(&lock);
    	printMatrix(&output->upper_matrix);
    	printMatrix(&output->lower_matrix);
        /*
    	printf("[");
    	for(int i=0;i<feature_range_length;i++){
    	    printf("%d ", feature_range[i]);
    	}
    	printf("] ");
    	if(isOverlap){
    	    printf("split_feature:%d isOverlap: True depth:%d\n", split_feature, depth);
    	}
    	else{
    	    printf("split_feature:%d isOverlap: False depth:%d\n", split_feature, depth);
    	}

    	printMatrix(&input->upper_matrix);
    	printMatrix(&input->lower_matrix);
        */
        if(isOverlap){
            printf("isOverlap: True depth:%d\n", depth);
        }
        else{
            printf("isOverlap: False depth:%d\n", depth);
        }
    	printf("\n");
        pthread_mutex_unlock(&lock);
    }

  
    if(depth==10 && isOverlap==0){
        pthread_mutex_lock(&lock);
            progress++;
            fprintf(stderr, "progress=%d/1024\n", progress);
            if(PARTIAL_MODE){
                denormalize_input_interval(nnet, input);
                printMatrix(&input->upper_matrix);
                printMatrix(&input->lower_matrix);
                printf("\n");
            }
        pthread_mutex_unlock(&lock);
        //printf("progress,%d/1024\n", progress);
    }
	
    if(isOverlap && NEED_FOR_ONE_RUN==0){
	isOverlap = split_interval(nnet, input, output, grad, depth,
                            feature_range, feature_range_length, split_feature,
                            equation_upper, equation_lower,
                            new_equation_upper, new_equation_lower);
    }
    else if (!isOverlap){
	pthread_mutex_lock(&lock);
	avg_depth -= (avg_depth) / AVG_WINDOW;
	avg_depth += depth / AVG_WINDOW;
	pthread_mutex_unlock(&lock);

    //printf("%f, %f\n", total_avg_depth,avg_depth);
#ifdef DEBUG
        pthread_mutex_lock(&lock);
        total_avg_depth = (total_avg_depth*leaf_num+depth)/(leaf_num+1);
        leaf_num++;
        if(depth>max_depth){
            max_depth = depth;
        }
        pthread_mutex_unlock(&lock);
        gettimeofday(&finish, NULL);

        float time_spent = ((float)(finish.tv_sec-start.tv_sec)*1000000 +\
                     (float)(finish.tv_usec-start.tv_usec)) / 1000000;
        float time_spent_last =  ((float)(last_finish.tv_sec-start.tv_sec)*1000000 +\
                     (float)(last_finish.tv_usec-start.tv_usec)) / 1000000;
        if((int)(time_spent)%20==0 && (time_spent - time_spent_last)>10){
            printf("progress: %f%%  total_avg_depth: %f  max_depth:%f avg_depth:%f\n",\
                     time_spent/pow(2, total_avg_depth+1)/0.000025/4*100, total_avg_depth,\
                     max_depth, avg_depth);
            gettimeofday(&last_finish, NULL);
        }
#endif
    }

	return isOverlap;
}

void check_adv(struct NNet* nnet, struct Interval *input){
    float a[nnet->inputSize];
    struct Matrix adv = {a, 1, nnet->inputSize};
    for(int i=0;i<nnet->inputSize;i++){
        float upper = input->upper_matrix.data[i];
        float lower = input->lower_matrix.data[i];
        float middle = (lower+upper)/2;
        a[i] = middle;
    }
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop(nnet, &adv, &output);
    int is_adv = 0;
    is_adv = check_functions1(nnet, &output);
    //printMatrix(&adv);
    //printMatrix(&output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(&adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
}

void check_adv1(struct NNet* nnet, struct Matrix *adv){
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop(nnet, adv, &output);
    int is_adv = 0;
    is_adv = check_functions1(nnet, &output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(&adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d\n", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
}


int split_interval(struct NNet *nnet, struct Interval *input,
               struct Interval *output, struct Interval *grad, 
               int depth, int *feature_range, int feature_range_length, 
               int split_feature, float *equation_upper, float *equation_lower,
               float *new_equation_upper, float *new_equation_lower){

    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize; 
    int maxLayerSize = nnet->maxLayerSize;
    float input_upper1[nnet->inputSize];
    float input_lower1[nnet->inputSize]; 
    float input_upper2[nnet->inputSize];
    float input_lower2[nnet->inputSize];
    
    pthread_mutex_lock(&lock);

    if(adv_found){
	pthread_mutex_unlock(&lock);
        return 0;
    }
	
    pthread_mutex_unlock(&lock);
   
    memcpy(input_upper1, input->upper_matrix.data, sizeof(float)*inputSize);
    memcpy(input_upper2, input->upper_matrix.data, sizeof(float)*inputSize);
    memcpy(input_lower1, input->lower_matrix.data, sizeof(float)*inputSize);
    memcpy(input_lower2, input->lower_matrix.data, sizeof(float)*inputSize);
    
    struct Interval input_interval1 = {
            (struct Matrix){input_lower1, 1, nnet->inputSize},
            (struct Matrix){input_upper1, 1, nnet->inputSize}
    };
    struct Interval input_interval2 = {
            (struct Matrix){input_lower2, 1, nnet->inputSize}, 
            (struct Matrix){input_upper2, 1, nnet->inputSize}
    };

    float o_upper1[nnet->outputSize], o_lower1[nnet->outputSize];
    struct Interval output_interval1 = {
            (struct Matrix){o_lower1, nnet->outputSize, 1},
            (struct Matrix){o_upper1, nnet->outputSize, 1}
    };
    float o_upper2[nnet->outputSize], o_lower2[nnet->outputSize];
    struct Interval output_interval2 = {
            (struct Matrix){o_lower2, nnet->outputSize, 1},
            (struct Matrix){o_upper2, nnet->outputSize, 1}
    };

    float grad_upper1[inputSize], grad_lower1[inputSize];
    struct Interval grad_interval1 = {
            (struct Matrix){grad_upper1, 1, nnet->inputSize},
            (struct Matrix){grad_lower1, 1, nnet->inputSize}
    };
    float grad_upper2[inputSize], grad_lower2[inputSize];
    struct Interval grad_interval2 = {
            (struct Matrix){grad_upper2, 1, nnet->inputSize},
            (struct Matrix){grad_lower2, 1, nnet->inputSize}
    };

    int feature_range1[feature_range_length];
    memcpy(feature_range1, feature_range, sizeof(int)*feature_range_length);
    int feature_range2[feature_range_length];
    memcpy(feature_range2, feature_range, sizeof(int)*feature_range_length);
    int feature_range_length1 = feature_range_length;
    int feature_range_length2 = feature_range_length;

    depth = depth + 1;

    //check mon
    int mono = 0;
    float smear = 0;
    float largest_smear = 0;
    for(int i=0;i<feature_range_length;i++){
        if(grad->upper_matrix.data[feature_range[i]]<=0 ||\
             grad->lower_matrix.data[feature_range[i]]>=0){
            mono = 1;
            #ifdef DEBUG
                printf("mono: %d ",i);  
            #endif    

            smear = ((grad->upper_matrix.data[feature_range[i]]>\
                -grad->lower_matrix.data[feature_range[i]])?\
                grad->upper_matrix.data[feature_range[i]]:\
                -grad->lower_matrix.data[feature_range[i]])*\
                (input->upper_matrix.data[feature_range[i]]-\
                input->lower_matrix.data[feature_range[i]]);

            if(smear>=largest_smear){
                largest_smear = smear;
                split_feature = i;
            }
            
        }
    }

    if(mono==1){
        #ifdef DEBUG
            printf("\nmono %d: %f %f\n", split_feature,\
                 grad->upper_matrix.data[feature_range[split_feature]],\
                 grad->lower_matrix.data[feature_range[split_feature]]);
        #endif
        feature_range_length1 = feature_range_length - 1;
        feature_range_length2 = feature_range_length - 1;
        for(int j=1;j<feature_range_length-split_feature;j++){
            feature_range1[split_feature+j-1] = feature_range[split_feature+j];
            feature_range2[split_feature+j-1] = feature_range[split_feature+j];
        }
        input_lower1[feature_range[split_feature]] =\
                     input_upper1[feature_range[split_feature]] =\
                     input->upper_matrix.data[feature_range[split_feature]];
        input_lower2[feature_range[split_feature]] =\
                     input_upper2[feature_range[split_feature]] =\
                     input->lower_matrix.data[feature_range[split_feature]];
        if(feature_range_length1 == 0){
            check_adv(nnet, &input_interval1);
            check_adv(nnet, &input_interval2);
            return 0;
        }
    }

    if(mono==0){
        float smear = 0;
        float largest_smear = 0;
        float smear_sum=0;
        float interval_range[feature_range_length];
        float e=0;
        for(int i=0;i<feature_range_length;i++){
            interval_range[i] = input->upper_matrix.data[feature_range[i]]-\
                                input->lower_matrix.data[feature_range[i]];
            e = (grad->upper_matrix.data[feature_range[i]]>\
                -grad->lower_matrix.data[feature_range[i]])?\
                grad->upper_matrix.data[feature_range[i]]:\
                -grad->lower_matrix.data[feature_range[i]];
            smear = e*interval_range[i];
            smear_sum += smear;
            if(largest_smear< smear){
                largest_smear = smear;
                split_feature = i;
            }
        }
        
        float upper = input->upper_matrix.data[feature_range[split_feature]];
        float lower = input->lower_matrix.data[feature_range[split_feature]];

        float middle;
        if(upper != lower){
            middle = (upper + lower) / 2;
        }
        else{
            middle = upper;
        }

        if(depth>=40){
            printMatrix(&input->upper_matrix);
            printMatrix(&input->lower_matrix);
        }

        if(smear_sum<=0.02){
            //printf("tighten:  smear_sum: %f  depth:%d  output:", smear_sum, depth);
            if(tighten_still_overlap(nnet, input, smear_sum)==0){
                //printf("0\n");
                pthread_mutex_lock(&lock);
                    smear_cnt ++;
                pthread_mutex_unlock(&lock);
                return 0;
            }
            //printf("1\n");
        }


        if(CHECK_ADV_MODE){
            if(depth >= 20 || upper-middle <= ADV_THRESHOLD){
                check_adv(nnet, input);
                return 0;
            }
        }
        else{
            if(depth >= 50 || upper-middle <= ADV_THRESHOLD){

#ifdef DEBUG
    if(depth>=50){
        printf("check for depth!\n"); 
    }
    else{
        printf("check for thershold\n");
    }
#endif
                check_adv(nnet, input);
            }
        }

        input_lower1[feature_range[split_feature]] = middle;
        input_upper2[feature_range[split_feature]] = middle;
    }

    //if(depth <= input_depth || ((count <= pow(2,(input_depth+1))-2-pow(2,(input_depth-1))) && depth <= avg[0]-7)){
    pthread_mutex_lock(&lock);
    if((depth <= avg_depth- MIN_DEPTH_PER_THREAD) && (count<=MAX_THREAD)) {
	pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        float *equation_upper1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
        float *equation_lower1 = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
        float *new_equation_upper1 = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
        float *new_equation_lower1 = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
        struct direct_run_check_args args1 = {nnet, &input_interval1,
                                         &output_interval1, &grad_interval1,
                                         depth, feature_range1,
                                         feature_range_length1, split_feature,
                                         equation_upper, equation_lower,
                                         new_equation_upper, new_equation_lower};

        struct direct_run_check_args args2 = {nnet, &input_interval2,
                                         &output_interval2, &grad_interval2,
                                         depth, feature_range2,
                                         feature_range_length2, split_feature,
                                         equation_upper1, equation_lower1,
                                         new_equation_upper1, new_equation_lower1};

        pthread_create(&workers1, NULL, direct_run_check_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
	thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL, direct_run_check_thread, &args2);
        pthread_mutex_lock(&lock);
        count++;
	thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_join(workers1, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);
        pthread_join(workers2, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);
        //printf ( "pid2: %ld done %d\n", syscall(SYS_gettid), count);

        free(equation_upper1);
        free(equation_lower1);
        free(new_equation_upper1);
        free(new_equation_lower1);

        if(depth==11){
            pthread_mutex_lock(&lock);
                progress++;
                fprintf(stderr, "progress=%d/1024 smear_cnt=%d\n",\
                        progress, smear_cnt );
                if(PARTIAL_MODE){
                    denormalize_input_interval(nnet, input);
                    printMatrix(&input->upper_matrix);
                    printMatrix(&input->lower_matrix);
                    printf("\n");
                }
            pthread_mutex_unlock(&lock);
        }
        return 0;
    }
    else{
	pthread_mutex_unlock(&lock);
        int isOverlap1 = direct_run_check(nnet, &input_interval1, 
                                     &output_interval1, &grad_interval1,
                                     depth, feature_range1, 
                                     feature_range_length1, split_feature,
                                     equation_upper, equation_lower,
                                     new_equation_upper, new_equation_lower);
        int isOverlap2 = direct_run_check(nnet, &input_interval2, 
                                     &output_interval2, &grad_interval2, 
                                     depth, feature_range2,
                                     feature_range_length2, split_feature,
                                     equation_upper, equation_lower,
                                     new_equation_upper, new_equation_lower);
        int result = isOverlap1 || isOverlap1;



        if(result==0 && depth==11){
        pthread_mutex_lock(&lock);
            progress++;
            fprintf(stderr, "progress=%d/1024 smear_cnt=%d\n", progress, smear_cnt );
            if(PARTIAL_MODE){
                denormalize_input_interval(nnet, input);
                printMatrix(&input->upper_matrix);
                printMatrix(&input->lower_matrix);
                printf("\n");
            }
        pthread_mutex_unlock(&lock);
        }

        return result;
    }
    
}


int pop_queue(int *wrong_nodes, int *wrong_node_length){
    if(*wrong_node_length==0){
        printf("underflow\n");
        can_t_prove = 1;
        return -1;
    }
    int node = wrong_nodes[0];
    for(int i=0;i<*wrong_node_length;i++){
        wrong_nodes[i] = wrong_nodes[i+1];
    }
    *wrong_node_length -= 1;
    return node;
}

int search_queue(int *wrong_nodes, int *wrong_node_length, int node_cnt){
    int wrong_ind=-1;
    for(int wn=0;wn<*wrong_node_length;wn++){
        if(node_cnt == wrong_nodes[wn]){
            wrong_ind = wn;
        }
    }
    return wrong_ind;
}

int forward_prop_interval_equation_lp(struct NNet *nnet, struct Interval *input,
                                 struct Interval *grad, int *output_map,
                                 float *equation_upper, float *equation_lower,
                                 float *new_equation_upper, float *new_equation_lower,
                                 int *wrong_nodes, int *wrong_node_length, int *sigs,
                                 float *wrong_up_s_up, float *wrong_up_s_low,
                                 float *wrong_low_s_up, float *wrong_low_s_low,
                                 int target, int sig,
                                 lprec *lp, int *rule_num)
{
    int i,j,k,layer;
    int node_cnt=0;
    int need_to_split = 0;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = inputSize;
    for(i=0;i<nnet->numLayers;i++){
        if(nnet->layerSizes[i]>maxLayerSize){
            maxLayerSize = nnet->layerSizes[i];
        }
    }

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer

    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    struct Interval equation_inteval = {
            (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
            (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
    };
    struct Interval new_equation_inteval = {
            (struct Matrix){(float*)new_equation_lower, inputSize+1, maxLayerSize},
            (struct Matrix){(float*)new_equation_upper, inputSize+1, maxLayerSize}
    };                                       

    float tempVal_upper=0.0, tempVal_lower=0.0;
    float upper_s_lower=0.0, lower_s_upper=0.0;

    float tempVal_upper1=0.0, tempVal_lower1=0.0;
    float upper_s_lower1=0.0, lower_s_upper1=0.0;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }
    float time_spent;

    for (layer = 0; layer<(numLayers); layer++)
    {
        
        memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        
        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];
        float p[weights.col*weights.row];
        float n[weights.col*weights.row];
        memset(p, 0, sizeof(float)*weights.col*weights.row);
        memset(n, 0, sizeof(float)*weights.col*weights.row);
        struct Matrix pos_weights = {p, weights.row, weights.col};
        struct Matrix neg_weights = {n, weights.row, weights.col};
        for(i=0;i<weights.row*weights.col;i++){
            if(weights.data[i]>=0){
                p[i] = weights.data[i];
            }
            else{
                n[i] = weights.data[i];
            }
        }

        matmul(&equation_inteval.upper_matrix, &pos_weights, &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &neg_weights, &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &pos_weights, &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &neg_weights, &new_equation_inteval.lower_matrix);
        
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];

            int wrong_ind = search_queue(wrong_nodes, wrong_node_length, node_cnt);

            if(node_cnt==target){
                if(sig==1){
                    set_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, sig, inputSize);
                }
                else{
                    set_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, sig, inputSize);
                    for(k=0;k<inputSize+1;k++){
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                    }
                }
                node_cnt ++;
                continue;
            }

            if(layer<(numLayers-1)){
                if(sigs[node_cnt] == 0 && node_cnt != target){
                    //printf("sigs0:%d\n", node_cnt);
                    for(k=0;k<inputSize+1;k++){
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                    }
                    node_cnt++;
                    continue;
                }

                if(sigs[node_cnt] == 1 && node_cnt != target){
                    //printf("sigs1:%d\n", node_cnt);
                    node_cnt++;
                    continue;
                }

                tempVal_upper = wrong_up_s_up[node_cnt];
                lower_s_upper = wrong_low_s_up[node_cnt];
                tempVal_lower = wrong_low_s_low[node_cnt];
                upper_s_lower = wrong_up_s_low[node_cnt];
                //printf("%d %d %d:%f %f %f %f\n", layer, i,node_cnt, tempVal_upper,upper_s_lower, lower_s_upper, tempVal_lower );
                if(wrong_ind!=-1){
                    /*nodes for update the equation*/
                    if(ACCURATE_MODE){
                        if(NEED_PRINT){
                            printf("%d %d %d:%f %f %f %f\n", layer, i, node_cnt, tempVal_upper, upper_s_lower, lower_s_upper, tempVal_lower );
                        }
                        if(set_wrong_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MAX, &tempVal_upper1)){
                            if(NEED_PRINT) printf("%d %d, uu, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MIN, &upper_s_lower1)){
                            if(NEED_PRINT) printf("%d %d, ul, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_lower, i*(inputSize+1), rule_num, inputSize, MAX, &lower_s_upper1)){
                            if(NEED_PRINT) printf("%d %d, lu, wrong\n",layer,i );
                            return 0;
                        }
                        if(set_wrong_node_constraints(lp, new_equation_lower, i*(inputSize+1), rule_num, inputSize, MIN, &tempVal_lower1)){
                            if(NEED_PRINT) printf("%d %d, ll, wrong\n",layer,i );
                            return 0;
                        }

                        if (tempVal_upper1<=0.0){
                            for(k=0;k<inputSize+1;k++){
                                new_equation_upper[k+i*(inputSize+1)] = 0;
                                new_equation_lower[k+i*(inputSize+1)] = 0;
                            }
                            R[layer][i] = 0;
                        }
                        else if(tempVal_lower1>=0.0){
                            R[layer][i] = 2;
                        }
                        else{
                            //printf("wrong node: ");
                            if(upper_s_lower1<0.0){
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_upper[k+i*(inputSize+1)] =\
                                                            new_equation_upper[k+i*(inputSize+1)]*\
                                                            tempVal_upper1 / (tempVal_upper1-upper_s_lower1);
                                }
                                new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper1*upper_s_lower1/\
                                                                    (tempVal_upper1-upper_s_lower1);
                            }

                            if(lower_s_upper1<0.0){
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_lower[k+i*(inputSize+1)] = 0;
                                }
                            }
                            else{
                                for(k=0;k<inputSize+1;k++){
                                    new_equation_lower[k+i*(inputSize+1)] =\
                                                            new_equation_lower[k+i*(inputSize+1)]*\
                                                            lower_s_upper1 / (lower_s_upper1- tempVal_lower1);
                                }
                            }
                            R[layer][i] = 1;
                        }
                        if(NEED_PRINT){
                            printf("%d %d %d:%f %f %f %f\n", layer, i, node_cnt, tempVal_upper1, upper_s_lower1, lower_s_upper1, tempVal_lower1 );
                        }
                    }
                    else{
                        for(k=0;k<inputSize+1;k++){
                            new_equation_upper[k+i*(inputSize+1)] =\
                                                    new_equation_upper[k+i*(inputSize+1)]*\
                                                    tempVal_upper / (tempVal_upper-tempVal_lower);
                        }
                        new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*tempVal_lower/\
                                                            (tempVal_upper-tempVal_lower);

                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] =\
                                                    new_equation_lower[k+i*(inputSize+1)]*\
                                                    tempVal_upper / (tempVal_upper- tempVal_lower);
                        }
                    }
                }
                else{
                    if(tempVal_upper<0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                            new_equation_upper[k+i*(inputSize+1)] = 0;
                        }
                    }
                }
            }
            else{
                if(i!=nnet->target){
                    //gettimeofday(&start, NULL);
                    float upper = 0.0;
                    float input_prev[inputSize];
                    struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                    memset(input_prev, 0, sizeof(float)*inputSize);
                    float o[outputSize];
                    struct Matrix output_matrix = {o, outputSize, 1};
                    memset(o, 0, sizeof(float)*outputSize);
                    if(output_map[i]){
                        if(!set_output_constraints(lp, new_equation_upper, i*(inputSize+1), rule_num, inputSize, MAX, &upper, input_prev)){
                            need_to_split = 1;
                            output_map[i] = 1;
                            if(NEED_PRINT){
                                printf("%d--Objective value: %f\n", i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        else{
                            output_map[i] = 0;
                            if(NEED_PRINT){
                                printf("%d--unsat\n", i);
                            }
                        }
                    }/*
                    gettimeofday(&finish, NULL);
                    time_spent = ((float)(finish.tv_sec-start.tv_sec)*1000000 +\
                        (float)(finish.tv_usec-start.tv_usec)) / 1000000;
                    printf("%d: %f \n",i, time_spent);
                    */
                }
            }
            node_cnt ++;
        }
        
        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
        
    }
    return need_to_split;
}


int direct_run_check_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     int target, int sig,
                     lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if(depth<=3){
        solve(lp);
    }

    int isOverlap = 0;

    isOverlap = forward_prop_interval_equation_lp(nnet, input,\
                                 grad, output_map,\
                                 equation_upper, equation_lower,\
                                 new_equation_upper, new_equation_lower,\
                                 wrong_nodes, wrong_node_length, sigs,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig,\
                                 lp, rule_num);

    if(depth<=PROGRESS_DEPTH && !isOverlap){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        pthread_mutex_unlock(&lock);
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            if(p>depth){
                total_progress[p-1] -= pow(2,(p-depth));
            }
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
    }

    if(isOverlap && !NEED_FOR_ONE_RUN){
        if(NEED_PRINT) printf("depth:%d, sig:%d Need to split!\n\n", depth, sig);
        isOverlap = split_interval_lp(nnet, input,\
                         grad, output_map,
                         equation_upper, equation_lower,\
                         new_equation_upper, new_equation_lower,\
                         wrong_nodes, wrong_node_length, sigs,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         lp, rule_num, depth);
    }
    else{
        if(!adv_found)
            if(NEED_PRINT) printf("depth:%d, sig:%d, UNSAT, great!\n\n", depth, sig);
            pthread_mutex_lock(&lock);
                avg_depth -= (avg_depth) / AVG_WINDOW;
                avg_depth += depth / AVG_WINDOW;
            pthread_mutex_unlock(&lock);
    }
    return isOverlap;
}

int split_interval_lp(struct NNet *nnet, struct Interval *input,
                     struct Interval *grad, int *output_map,
                     float *equation_upper, float *equation_lower,
                     float *new_equation_upper, float *new_equation_lower,
                     int *wrong_nodes, int *wrong_node_length, int *sigs,
                     float *wrong_up_s_up, float *wrong_up_s_low,
                     float *wrong_low_s_up, float *wrong_low_s_low,
                     lprec *lp, int *rule_num, int depth)
{
    pthread_mutex_lock(&lock);
    if(adv_found){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    
    if(depth>=15){
        printf("depth %d too deep!\n", depth );
        can_t_prove = 1;
    }

    if(can_t_prove){
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if(depth==0){
        memset(progress_list, 0, PROGRESS_DEPTH*sizeof(int));
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            total_progress[p-1] = pow(2, p);
        }
    }

    depth ++;

    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;
    int outputSize = nnet->outputSize;

    int target = 0;
    target = pop_queue(wrong_nodes, wrong_node_length);
    int sig = 0;
    int isOverlap1 = 0;
    int isOverlap2 = 0;

    float *equation_upper1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *equation_lower1 = (float*)malloc(sizeof(float) *\
                            (inputSize+1)*maxLayerSize);
    float *new_equation_upper1 = (float*)malloc(sizeof(float) *\
                                (inputSize+1)*maxLayerSize);
    float *new_equation_lower1 = (float*)malloc(sizeof(float) *\
                                    (inputSize+1)*maxLayerSize);
    lprec *lp1, *lp2;
    //write_lp(lp, "model.lp");
    //lp1 = read_LP("model.lp", IMPORTANT, NULL);
    //lp2 = read_LP("model.lp", IMPORTANT, NULL);
    lp1 = copy_lp(lp);
    lp2 = copy_lp(lp);

    int rule_num1 = *rule_num;
    int rule_num2 = *rule_num;

    int wrong_node_length1 = *wrong_node_length;
    int wrong_node_length2 = *wrong_node_length; 
    int wrong_nodes1[wrong_node_length1];
    int wrong_nodes2[wrong_node_length2];
    memcpy(wrong_nodes1, wrong_nodes, sizeof(int)*wrong_node_length1);
    memcpy(wrong_nodes2, wrong_nodes, sizeof(int)*wrong_node_length2);

    int output_map1[outputSize];
    int output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(int)*outputSize);
    memcpy(output_map2, output_map, sizeof(int)*outputSize);

    int sigSize = 0; 
    for(int layer=1;layer<nnet->numLayers;layer++){
        sigSize += nnet->layerSizes[layer];
    }

    int sigs1[sigSize];
    int sigs2[sigSize];

    memcpy(sigs1, sigs, sizeof(int)*sigSize);
    memcpy(sigs2, sigs, sizeof(int)*sigSize);

    int sig1,sig2;
    sig1 = 1;
    sig2 = 0;
    sigs1[target] = 1;
    sigs2[target] = 0;
    pthread_mutex_lock(&lock);
    if((depth <= avg_depth- MIN_DEPTH_PER_THREAD) && (count<=MAX_THREAD)) {
        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        struct direct_run_check_lp_args args1 = {nnet, input,\
                                 grad, output_map1,
                                 equation_upper, equation_lower,\
                                 new_equation_upper, new_equation_lower,\
                                 wrong_nodes1, &wrong_node_length1, sigs1,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig1,\
                                 lp1, &rule_num1, depth};

        struct direct_run_check_lp_args args2 = {nnet, input,\
                                 grad, output_map2,
                                 equation_upper1, equation_lower1,\
                                 new_equation_upper1, new_equation_lower1,\
                                 wrong_nodes2, &wrong_node_length2, sigs2,\
                                 wrong_up_s_up, wrong_up_s_low,\
                                 wrong_low_s_up, wrong_low_s_low,\
                                 target, sig2,\
                                 lp2, &rule_num2, depth};

        pthread_create(&workers1, NULL, direct_run_check_lp_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, NULL, direct_run_check_lp_thread, &args2);
        pthread_mutex_lock(&lock);
        count++;
        thread_tot_cnt++;
        pthread_mutex_unlock(&lock);
        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_join(workers1, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);
        pthread_join(workers2, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

    }
    else{
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_lp(nnet, input,\
                         grad, output_map1,
                         equation_upper, equation_lower,\
                         new_equation_upper, new_equation_lower,\
                         wrong_nodes1, &wrong_node_length1, sigs1,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         target, sig1,\
                         lp1, &rule_num1, depth);

        isOverlap2 = direct_run_check_lp(nnet, input,\
                         grad, output_map2,
                         equation_upper1, equation_lower1,\
                         new_equation_upper1, new_equation_lower1,\
                         wrong_nodes2, &wrong_node_length2, sigs2,\
                         wrong_up_s_up, wrong_up_s_low,\
                         wrong_low_s_up, wrong_low_s_low,\
                         target, sig2,\
                         lp2, &rule_num2, depth);
    }

    free(equation_upper1);
    free(equation_lower1);
    free(new_equation_upper1);
    free(new_equation_lower1);
    delete_lp(lp1);
    delete_lp(lp2);

    int result = isOverlap1 || isOverlap2;
    depth --;

    if(!result && depth<=PROGRESS_DEPTH){
        pthread_mutex_lock(&lock);
            progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            fprintf(stderr, " %d/%d ", progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}
