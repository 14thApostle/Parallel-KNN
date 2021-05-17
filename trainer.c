#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

char **str_split(char *a_str, const char a_delim)
{
    char **result = 0;
    size_t count = 0;
    char *tmp = a_str;
    char *last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char *) * count);

    if (result)
    {
        size_t idx = 0;
        char *token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

int main(int argc, char *argv[])
{

    FILE *fp1;
    char path[1035];

    // Find number of training examples
    int num_train_examples;
    fp1 = popen("/bin/ls ./datasets/food_1_ftrs/train/ | wc -l", "r");
    if (fp1 == NULL)
    {
        printf("Failed to run command\n");
        exit(1);
    }
    fgets(path, sizeof(path), fp1);
    num_train_examples = atoi(path);
    printf("%d Training examples \n", num_train_examples);

    // Train files list
    fp1 = popen("/bin/ls ./datasets/food_1_ftrs/train/", "r");
    if (fp1 == NULL)
    {
        printf("Failed to run command\n");
        exit(1);
    }
    char **files_list = (char **)malloc(sizeof(char *) * num_train_examples);
    int i = 0;
    while (fgets(path, sizeof(path), fp1) != NULL)
    {
        path[strcspn(path, "\n")] = 0;
        char full_path[1035] = "./datasets/food_1_ftrs/train/";
        strcat(full_path, path);
        int str_len = strlen(full_path);
        files_list[i] = malloc(sizeof(char) * str_len);
        strcpy(files_list[i], full_path);
        i += 1;
    }
    pclose(fp1);

    /*
    TODO

    - Init MPI and perform a Scatter here of (files_list) (which is the list of all training files)
    - Divide work and each threads read a block of files. 
    */

    // Read file contents
    FILE *fp;
    char line1[1023];
    char line2[10000];

    printf("The %d train file: %s\n", 0, files_list[0]);
    fp = fopen(files_list[0], "r");
    if (fp == NULL)
    {
        printf("Fail");
        exit(EXIT_FAILURE);
    }

    fgets(line1, sizeof(line1), fp);
    line1[strcspn(line1, "\n")] = 0;
    fgets(line2, sizeof(line2), fp);

    // Headers are some information stored in the txt file
    // Format (image_id, class_name, len_feature_vector)
    char **headers;
    headers = str_split(line1, ' ');
    for (int i = 0; i < 3; i++)
    {
        printf("The %d header is : %s\n", i, headers[i]);
    }

    char **t_vectors;
    t_vectors = str_split(line2, ' ');
    double ftr_vectors[10000];
    for (int i = 0; i < atoi(headers[2]); i++)
    {
        ftr_vectors[i] = strtod(t_vectors[i], NULL);
        printf("The %dth vector val is : %f\n", i, ftr_vectors[i]);
    }
    /*
    TODO

    - We have the array ftr_vectors to train on. (length = headers[2])
    - Create our KNN instance and perform initialization
    - Write update rules for each iteration, Gather all outputs before tho

    */

    fclose(fp);
    return 0;
}