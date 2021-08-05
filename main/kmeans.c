#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#define K 2
#define threshold 1e-8

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

double distance_fn(double pointa[], double pointb[], int n)
{
    double dist = 0;
    for (int i = 0; i < n; i++)
    {
        dist += pow(pointa[i] - pointb[i], 2);
    }
    dist = sqrt(dist);
    return dist;
}

int min_distance_centroid(double centroids[][128], double point[])
{
    double min_dist = distance_fn(centroids[0], point, 128);
    int min_dist_centroid = 0;
    double dist;

    // printf("Dist = %f for centroid %d\n", min_dist, 0);
    for (int i = 1; i < K; i++)
    {
        dist = distance_fn(centroids[i], point, 128);
        // printf("Dist = %f for centroid %d\n", dist, i);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_dist_centroid = i;
        }
    }
    return min_dist_centroid;
}

void save_centroid_values(double centroids[K][128])
{
    FILE *fp;
    fp = fopen("centroids.txt", "w");
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            fprintf(fp, "%f ", centroids[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
    int myid, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);   /* myrank of the process */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* size of the communicator */

    printf("Process id %d\n", myid);
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

    // Initialize centroids
    double centroids[K][128];
    if (myid == 0)
    {
        for (int i = 0; i < K; i++)
        {
            // Read file contents
            char *file_path = files_list[i];
            FILE *fp;
            char line1[1023];
            char line2[10000];

            // printf("The %d train file: %s\n", 0, file_path);
            fp = fopen(file_path, "r");
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

            char **t_vectors;
            t_vectors = str_split(line2, ' ');
            double ftr_vectors[10000];
            for (int j = 0; j < atoi(headers[2]); j++)
            {
                ftr_vectors[j] = strtod(t_vectors[j], NULL);
                centroids[i][j] = ftr_vectors[j];
            }
            fclose(fp);
        }
    }

    // Find closest centroid for each datapoint and
    // find the new centroids by averaging feature vectors
    // of classifications
    for (int iterations = 0; iterations < 10; iterations++)
    {
        // Broadcast centroids to all processes
        for (int i = 0; i < K; i++)
        {
            MPI_Bcast(&centroids[i], 128, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        double local_new_centroids[K][128];
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 128; j++)
            {
                local_new_centroids[i][j] = 0;
            }
        }
        int local_new_example_centroids[K] = {0};

        int q = num_train_examples / nprocs;
        int start = myid * q;
        for (int i = start; i < start + q; i++)
        {
            // Read file contents
            char *file_path = files_list[i];
            FILE *fp;
            char line1[1023];
            char line2[10000];

            // printf("The %d train file: %s\n", 0, file_path);
            fp = fopen(file_path, "r");
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
            // for (int j = 0; j < 3; j++)
            // {
            //     printf("The %d header is : %s\n", j, headers[j]);
            // }

            char **t_vectors;
            t_vectors = str_split(line2, ' ');
            double ftr_vectors[10000];
            for (int j = 0; j < atoi(headers[2]); j++)
            {
                ftr_vectors[j] = strtod(t_vectors[j], NULL);
            }
            int centroid_id = min_distance_centroid(centroids, ftr_vectors);
            local_new_example_centroids[centroid_id] += 1;
            for (int j = 0; j < 128; j++)
            {
                local_new_centroids[centroid_id][j] += ftr_vectors[j];
            }
            // printf("Closest centroid is %d\n", centroid_id);
            fclose(fp);
        }

        double new_centroids[K][128];
        if (myid == 0)
        {
            for (int i = 0; i < K; i++)
                for (int j = 0; j < 128; j++)
                    new_centroids[i][j] = 0;
        }

        for (int i = 0; i < K; i++)
        {
            MPI_Reduce(local_new_centroids[i], new_centroids[i], 128, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        int new_example_centroids[K] = {0};

        MPI_Reduce(local_new_example_centroids, new_example_centroids, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        int done = 0;
        if (myid == 0)
        {

            for (int i = 0; i < K; i++)
                for (int j = 0; j < 128; j++)
                    new_centroids[i][j] /= new_example_centroids[i];

            // Find the error values
            double error = 0;
            for (int i = 0; i < K; i++)
            {
                double l_error = distance_fn(new_centroids[i], centroids[i], 128);
                if (l_error >= error)
                {
                    error = l_error;
                }
            }
            printf("Error after %d is %f\n", iterations, error);

            // Update the centroids
            for (int i = 0; i < K; i++)
                for (int j = 0; j < 128; j++)
                    centroids[i][j] = new_centroids[i][j];

            if (error < threshold)
            {
                done = 1;
            }
        }
        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (done)
        {
            break;
        }
    }

    save_centroid_values(centroids);

    MPI_Finalize();
    return 0;
}