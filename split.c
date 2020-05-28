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
#define MAX_RUNTIME 3600 // 1 hour

int NEED_PRINT = 0;
int NEED_FOR_ONE_RUN = 0;
int input_depth = 0;
bool adv_found = false;
bool analysis_uncertain = false;
int count = 0;
int thread_tot_cnt  = 0;
int smear_cnt = 0;

int progress = 0;
int MAX_DEPTH = 1215752190; // never abort due to depth

int progress_list[PROGRESS_DEPTH];
int total_progress[PROGRESS_DEPTH];

int analyses_count = 0;


 /*
  * You need to customize your own checking function.
  * Here is a couple of sample functions that you could use.
  */
bool check_not_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_max_constant(struct NNet *nnet, struct Interval *output){
    if(output->upper_matrix.data[nnet->target]>0.5011){
        return true;
    }
    else{
        return false;
    }
}

bool check_max(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->lower_matrix.data[i]>0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_min(struct NNet *nnet, struct Interval *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]<0 && i != nnet->target){
            return false;
        }
    }
    return true;
}

bool check_not_min(struct NNet *nnet, struct Interval *output){
	for(int i=0;i<nnet->outputSize;i++){
		if(output->lower_matrix.data[i]<0 && i != nnet->target){
			return true;
		}
	}
	return false;
}


bool check_not_min_p11(struct NNet *nnet, struct Interval *output){

    if(output->lower_matrix.data[0]<0)
        return true;

    return false;
}


bool check_max_constant1(struct NNet *nnet, struct Matrix *output){
    if(output->data[nnet->target]<0.5011){
        return false;
    }
    return true;
}


bool check_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return false;
        }
    }
    return true;
}


bool check_not_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_not_min1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]<0 && i != nnet->target){
            return true;
        }
    }
    return false;
}


bool check_not_max_norm(struct NNet *nnet, struct Interval *output){
    double t = output->lower_matrix.data[nnet->target];
    for(int i=0;i<nnet->outputSize;i++){
        if(output->upper_matrix.data[i]>t && i != nnet->target){
            return true;
        }
    }
    return false;
}


/*
 * Here is the function checking whether the output range always 
 * satisfies your customized safety property.
 */
bool check_functions(struct NNet *nnet, struct Interval *output){
    if (PROPERTY == 1){
        /*
         * You need to customize your own checking function
         * For instance, you can check whether the first output
         * is always the smallest. You can also check whether 
         * one of the output is always larger than 0.001, etc.
         */
    }

    return check_not_max(nnet, output);
}


/*
 * Here is the function checking whether the output range always 
 * satisfies your customized safety property but without output norm.
 * This is only used in network_test.c once for checking before splits.
 */
bool check_functions_norm(struct NNet *nnet, struct Interval *output){
    return check_not_max_norm(nnet, output);
}


/*
 * Here is the function checking whether the given concrete outupt 
 * violates your customized safety property.
 */
bool check_functions1(struct NNet *nnet, struct Matrix *output){

    if (PROPERTY == 1){
        /*
         * You need to customize your own checking function for adv
         * For instance, you can check whether the first output
         * is always the smallest. You can also check whether 
         * one of the output is always larger than 0.001, etc.
         */
    }

    return check_not_max1(nnet, output);
}










/*
 * Multithread function
 */
void *direct_run_check_conv_lp_thread(void *args){
    struct direct_run_check_conv_lp_args *actual_args = args;
    direct_run_check_conv_lp(actual_args->nnet,
        actual_args->input,
        actual_args->output_map,
        actual_args->grad,
        actual_args->sigs,
        actual_args->target,
        actual_args->lp, 
        actual_args->rule_num,
        actual_args->depth,
        actual_args->start_time,
        actual_args->increment_global_counter);
    return NULL;
}


void check_adv1(struct NNet* nnet, struct Matrix *adv){
    double out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop_conv(nnet, adv, &output);
    bool is_adv = check_functions1(nnet, &output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(adv);
        printMatrix(&output);
        int adv_output = nnet->target;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>output.data[adv_output]){
                adv_output = i;
            }
        }
        printf("%d ---> %d\n", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = true;

        evaluate_conv(nnet, adv, &output);
        printf("Original adv: ");
        printMatrix(&output);
        for (int i = 0; i < nnet->inputSize; i++)
        {
            if(adv->data[i] < nnet->input_interval->lower_matrix.data[i] - 0.0000001 || 
                adv->data[i] > nnet->input_interval->upper_matrix.data[i] + 0.0000001) {
                printf("Invalid input (%d): %.20f < %.20f < %.20f not valid \n",
                    i, nnet->input_interval->lower_matrix.data[i], adv->data[i], nnet->input_interval->upper_matrix.data[i]);
                //exit(1);
            }
        }
        pthread_mutex_unlock(&lock);
    }
}


int pop_queue(int *wrong_nodes, int *wrong_node_length){
    if(*wrong_node_length==0){
        printf("underflow\n");
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

int max(double a, double b){
    return (a>b)?a:b;
}

int min(double a, double b){
    return (a<b)?a:b;
}


int sym_relu_lp(struct Interval *input,
                    struct NNet *nnet,
                    int R[][nnet->maxLayerSize],
                    int layer, int *err_row,
                    int *wrong_nodes_map, 
                    int*wrong_node_length, int *node_cnt,
                    int target, int *sigs,
                    lprec *lp, int *rule_num){
    
    //record the number of wrong nodes
    int wcnt = 0;

    int inputSize = nnet->inputSize;

    update_equations(nnet, layer, false);


    for (int i=0; i < nnet->layerSizes[layer+1]; i++)
    {

        double low_tempVal_upper = 0;
        double low_tempVal_lower = 0;
        double up_tempVal_upper = 0;
        double up_tempVal_lower = 0;

        relu_bound(nnet, input, i, layer, *err_row, &up_tempVal_lower, &up_tempVal_upper, 2, false);
        relu_bound(nnet, input, i, layer, *err_row, &low_tempVal_lower, &low_tempVal_upper, 1, false);


        if(*node_cnt == target){
            if(sigs[target]==1){
                set_node_constraints(lp, &(nnet->cache_equation_low[i*(inputSize)]), nnet->cache_bias_low[i], \
                        0, rule_num, 1, inputSize);
                //set_node_constraints(lp, (*new_sInterval->matrix_up).data,
                //        i*(inputSize+1), rule_num, sigs[target], inputSize);
            }
            else{
                set_node_constraints(lp, &(nnet->cache_equation_up[i*(inputSize)]), nnet->cache_bias_up[i], \
                        0, rule_num, 0, inputSize);
                //set_node_constraints(lp, (*new_sInterval->matrix_low).data,
                //        i*(inputSize+1), rule_num, sigs[target], inputSize);
            }
        }
        double low_temp_upper_gt = 0;
        double low_temp_lower_gt = 0;
        relu_bound(nnet, input, i, layer, *err_row, &low_temp_lower_gt, &low_temp_upper_gt, 1, true);

        // handle the nodes that are split
        if(sigs[*node_cnt] == 0){
            low_tempVal_upper = 0;
            up_tempVal_upper = 0;
            if(low_tempVal_upper > 0) {
                low_tempVal_upper = 0;
            }
            if(low_tempVal_lower > 0) {
                low_tempVal_lower = 0;
            }
            if(up_tempVal_upper > 0) {
                up_tempVal_upper = 0;
            }
            if(up_tempVal_lower > 0) {
                up_tempVal_lower = 0;
            }
            if(low_temp_lower_gt > 0){
                low_temp_lower_gt = 0;
            }
        }
        else if(sigs[*node_cnt] == 1){
            low_tempVal_lower = 0;
            up_tempVal_lower = 0;
            if(low_tempVal_lower < 0) {
                low_tempVal_lower = 0;
            }
            if(low_tempVal_upper < 0) {
                low_tempVal_upper = 0;
            }
            if(up_tempVal_lower < 0) {
                up_tempVal_lower = 0;
            }
            if(up_tempVal_upper < 0) {
                up_tempVal_upper = 0;
            }
            if(low_temp_lower_gt < 0) {
                low_temp_lower_gt = 0;
            }
        }

        //Perform ReLU relaxation
        if(low_tempVal_lower > low_tempVal_upper || up_tempVal_lower > up_tempVal_upper) {
            printf("2) Invalid bounds \n");
            exit(1);
        }

        if(low_tempVal_lower > up_tempVal_lower || low_tempVal_upper > up_tempVal_upper) {
            printf("2) Invalid (switched) bounds \n");
            printf("(%.20f - %.20f) - (%.20f - %.20f) \n", low_tempVal_lower, low_tempVal_upper, up_tempVal_lower, up_tempVal_upper);
            //printf("Upper: "); printMatrix(sym_interval->matrix_up);
            //printf("Lower: "); printMatrix(sym_interval->matrix_low);
            //printf("Error: "); printMatrix(sym_interval->err_matrix);
            exit(1);
        }
    
        //printf("Before relax UP (%f - %f), LOW (%f - %f) \n",
        //    up_tempVal_lower, up_tempVal_upper, low_tempVal_lower, low_tempVal_upper);

            
        R[layer][i] = relax_relu(nnet, low_tempVal_lower, low_tempVal_upper,
            up_tempVal_lower, up_tempVal_upper, low_temp_lower_gt, input, i, layer,
            err_row, wrong_node_length, &wcnt, true);

        //double tempVal_upper=0.0, tempVal_lower=0.0;
        //relu_bound(nnet, input, i, layer, *err_row,
        //            &tempVal_lower, &tempVal_upper, 0);

        //printf("After ReLu: Layer %d, node %d: %f - %f \n", layer, i, tempVal_lower, tempVal_upper);


        if(R[layer][i] >= 10) {
            wrong_nodes_map[(*wrong_node_length) - 1] = *node_cnt;
            R[layer][i] = 1;
        }

        *node_cnt += 1;
    }

    nnet->cache_valid = false;
    nnet->cache_valid_gtzero = false;

    return wcnt;
}


bool forward_prop_interval_equation_conv_lp(struct NNet *nnet,
                         struct Interval *input, bool *output_map, double *grad,
                         int *wrong_nodes_map, int *wrong_node_length,
                         int *sigs, int target, lprec *lp, int *rule_num)
{
    int node_cnt=0;
    bool need_to_split = false;
    int maxLayerSize   = nnet->maxLayerSize;

    int numLayers    = nnet->numLayers;
    int outputSize   = nnet->outputSize;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(int)*numLayers*maxLayerSize);

    //err_row is the number that is wrong before current layer
    int err_row=0;
    for (int layer = 0; layer<numLayers; layer++)
    {
        //printf("sig:%d, layer:%d\n",sig, layer );
        
        
        if (CHECK_ADV_MODE){
            printf("Not implemented! \n");
        }
        else{
            if(nnet->layerTypes[layer] == 0){
                //printf("fc layer");
                //sym_fc_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            }
            else{
                printf("Conv not implemented \n");
                exit(1);
                //printf("conv layer\n");
                //sym_conv_layer(&sInterval, &new_sInterval, nnet, layer, err_row);
            }
        }
        
        if(layer<(numLayers-1)){
            // printf("relu layer\n");
            sym_relu_lp(input, nnet, R, layer,\
                        &err_row, wrong_nodes_map, wrong_node_length, &node_cnt,\
                        target, sigs, lp, rule_num);
        }
        else{
            int inputSize = nnet->inputSize;

            //get_equations(nnet, layer, equation_low, equation_up, bias_low, bias_up);

            for (int i=0; i < nnet->layerSizes[layer+1]; i++){
                if(nnet->weights_up[nnet->numLayers].col != 1) {
                    printf("Too many last nodes \n");
                    exit(1);
                }
                memset(nnet->weights_up[nnet->numLayers].data, 0, sizeof(double)*nnet->weights_up[nnet->numLayers].col*nnet->weights_up[nnet->numLayers].row);
                nnet->weights_up[nnet->numLayers].data[i] = 1;
                nnet->weights_up[nnet->numLayers].data[nnet->target] = -1;
                nnet->cache_valid = false;
                nnet->cache_valid_gtzero = false;

                if(NEED_PRINT){
                    double tempVal_upper=0.0, tempVal_lower=0.0;
                    relu_bound(nnet, input, i, layer, err_row,
                            &tempVal_lower, &tempVal_upper, 0, false);
                    //printf("target:%d, sig:%d, node:%d, l:%f, u:%f\n",
                    //            target, sigs[target], i, tempVal_lower, tempVal_upper);
                    printf("After ReLu: Layer %d, node %d: %f - %f \n", layer, i, tempVal_lower, tempVal_upper);
                }
                

                if(i!=nnet->target){
                    //gettimeofday(&start, NULL);
                    double upper = 0.0;
                    double input_prev[inputSize];
                    struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                    memset(input_prev, 0, sizeof(double)*inputSize);
                    double o[outputSize];
                    memset(o, 0, sizeof(double)*outputSize);
                    if(output_map[i]){
                        update_equations(nnet, layer+1, false);

                        int search = set_output_constraints(lp, &(nnet->cache_equation_up[0*(inputSize)]), nnet->cache_bias_up[0],
                            0, rule_num, inputSize, MAX, &upper,
                            input_prev);
                        if(search == 1){
                            need_to_split = true;
                            output_map[i] = true;
                            if(NEED_PRINT){
                                printf("target:%d, sig:%d, node:%d--Objective value: %f\n",\
                                            target, sigs[target], i, upper);
                            }
                            check_adv1(nnet, &input_prev_matrix);
                            if(adv_found){
                                return 0;
                            }
                        }
                        else if(search == -1)  { // timeout
                            need_to_split = 1;
                        }
                        else{
                            output_map[i] = false;
                            if(NEED_PRINT){
                                printf("target:%d, sig:%d, node:%d--unsat\n",\
                                            target, sigs[target], i);
                            }
                        }
                    }

                }

                node_cnt++;
            }
        }
    
        if(err_row >= ERR_NODE) {
            printf("err_row = %d > %d \n", err_row, ERR_NODE);
            exit(1);
        }
    }

    //printf("sig:%d, need_to_split:%d\n",sig, need_to_split );
    int total_nodes = 0;
    for(int layer=1;layer<numLayers;layer++){
        total_nodes += nnet->layerSizes[layer];
    }
    memset(grad, 0, sizeof(float)*total_nodes);
    backward_prop_conv(nnet, grad, R);

    return need_to_split;
}


int direct_run_check_conv_lp(struct NNet *nnet, struct Interval *input,
    bool *output_map, double *grad, int *sigs, int target, lprec *lp,
    int *rule_num, int depth, struct timeval start_time,
    bool increment_global_counter)
{
    if(adv_found){
        return 0;
    }

    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    if(current_time.tv_sec - start_time.tv_sec > MAX_RUNTIME) {
        // Print the error message only once (thread safety not required)
        if(!analysis_uncertain) {
            printf("Timeout during splits \n");
        }
        analysis_uncertain = true;
        return 0;
    }

    int sub_analyses = 0;

    if(depth<=3){
        solve(lp);
    }

    int total_nodes = 0;
    for(int layer=1;layer<nnet->numLayers;layer++){
        total_nodes += nnet->layerSizes[layer];
    }
    int *wrong_nodes_map = (int*)SAFEMALLOC(sizeof(int) * total_nodes);
    memset(wrong_nodes_map,0,sizeof(int)*total_nodes);
    int wrong_node_length = 0;

    bool isOverlap = forward_prop_interval_equation_conv_lp(nnet, input,\
                            output_map, grad, wrong_nodes_map, &wrong_node_length, \
                            sigs, target, lp, rule_num);

    //printf("sig:%d, i:%d\n",sig, isOverlap );
    if(depth<=PROGRESS_DEPTH && !isOverlap){
        progress_list[depth-1] += 1;
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
        if(NEED_PRINT)
            printf("depth:%d, sig:%d Need to split!\n\n", depth, sigs[target]);
        sub_analyses = split_interval_conv_lp(nnet, input, output_map, grad,
                         wrong_nodes_map, &wrong_node_length, sigs,
                         lp, rule_num, depth, start_time, false);
    }
    else{
        if(!adv_found)
            if(NEED_PRINT) 
                printf("depth:%d, sig:%d, UNSAT, great!\n\n", depth, sigs[target]);
    }

    free(wrong_nodes_map);
    
    // Undo last split
    del_constraint(lp, *rule_num);
    *rule_num -= 1; 

    if(increment_global_counter) {
        pthread_mutex_lock(&lock);
        analyses_count += 1 + sub_analyses;
        pthread_mutex_unlock(&lock);
    }

    return 1 + sub_analyses;
}


int split_interval_conv_lp(struct NNet *nnet, struct Interval *input,
    bool *output_map, double *grad, int *wrong_nodes, int *wrong_node_length,
    int *sigs, lprec *lp, int *rule_num, int depth, struct timeval start_time,
    bool increment_global_counter)
{
    if(adv_found){
        return 0;
    }
    
    if(depth>=MAX_DEPTH){
        printf("Maximum depth reached\n");
        analysis_uncertain = true;
        return 0;
    }


    if(depth==0){
        memset(progress_list, 0, PROGRESS_DEPTH*sizeof(int));
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            total_progress[p-1] = pow(2, p);
        }
    }

    depth ++;

    int sub_analyses = 0;

    int outputSize = nnet->outputSize;

    sort(grad, *wrong_node_length, wrong_nodes);
    sort_layers(nnet->numLayers, nnet->layerSizes,\
            *wrong_node_length, wrong_nodes);
    int target = pop_queue(wrong_nodes, wrong_node_length);
    if(target == -1) {
        pthread_mutex_lock(&lock);
        analysis_uncertain = true;
        pthread_mutex_unlock(&lock);
        return 0;
    }
    // printf("%d, %d\n", wrong_nodes[0], wrong_nodes[1]);
    

    int rule_num1 = *rule_num;
    int rule_num2 = *rule_num;

    int total_nodes = 0; 
    for(int layer=1;layer<nnet->numLayers;layer++){
        total_nodes += nnet->layerSizes[layer];
    }

    bool output_map1[outputSize];
    bool output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(bool)*outputSize);
    memcpy(output_map2, output_map, sizeof(bool)*outputSize);


    int *sigs1 = (int*)SAFEMALLOC(sizeof(int) * total_nodes);
    int *sigs2 = (int*)SAFEMALLOC(sizeof(int) * total_nodes);

    memcpy(sigs1, sigs, sizeof(int)*total_nodes);
    memcpy(sigs2, sigs, sizeof(int)*total_nodes);

    sigs1[target] = 1;
    sigs2[target] = 0;


    if(count<MAX_THREAD && !NEED_FOR_ONE_RUN) {
        lprec *lp1, *lp2;
        lp1 = lp;
        lp2 = copy_lp(lp);
        reset_conv_network(nnet);
        struct NNet *nnet1 = nnet;
        struct NNet *nnet2 = duplicate_conv_network(nnet);

        pthread_attr_t attr;
        int status = pthread_attr_init(&attr);
        if (status != 0) {
            printf("pthread_attr_init: %d \n", status);
            exit(1);
        }
        status = pthread_attr_setstacksize(&attr, 8*1024*1024);
        if (status != 0) {
            printf("pthread_attr_setstacksize: %d \n", status);
            exit(1);
        }


        pthread_t workers1, workers2;
        struct direct_run_check_conv_lp_args args1 = {
                            nnet1, input, output_map1, grad,
                            sigs1,\
                            target, lp1, &rule_num1, depth, start_time, true
                        };

        struct direct_run_check_conv_lp_args args2 = {
                            nnet2, input, output_map2, grad,
                            sigs2,\
                            target, lp2, &rule_num2, depth, start_time, true
                        };

        pthread_create(&workers1, &attr,\
                direct_run_check_conv_lp_thread, &args1);
        pthread_mutex_lock(&lock);
        count += 2;
        thread_tot_cnt += 2;
        pthread_mutex_unlock(&lock);
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        pthread_create(&workers2, &attr,\
                direct_run_check_conv_lp_thread, &args2);
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

        delete_lp(lp2);
        destroy_conv_network(nnet2);
    }
    else{
        lprec *lp1, *lp2;
        lp1 = lp;
        reset_conv_network(nnet);
        struct NNet *nnet1 = nnet;
        sub_analyses += direct_run_check_conv_lp(nnet1, input,\
                            output_map1, grad,\
                            sigs1,\
                            target, lp1, &rule_num1, depth, start_time, false);

        lp2 = lp;
        reset_conv_network(nnet);
        struct NNet *nnet2 = nnet;
        sub_analyses += direct_run_check_conv_lp(nnet2, input,\
                            output_map2, grad,\
                            sigs2,\
                            target, lp2, &rule_num2, depth, start_time, false);
    }

    depth --;

    if(depth<=PROGRESS_DEPTH){
        progress_list[depth-1] += 1;
        fprintf(stderr, " progress: ");
        for(int p=1;p<PROGRESS_DEPTH+1;p++){
            fprintf(stderr, " %d/%d ",\
                    progress_list[p-1], total_progress[p-1]);
        }
        fprintf(stderr, "\n");
    }

    if(increment_global_counter) {
        pthread_mutex_lock(&lock);
        analyses_count += sub_analyses;
        pthread_mutex_unlock(&lock);
    }

    free(sigs1);
    free(sigs2);

    return sub_analyses;
}
