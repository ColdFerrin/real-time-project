// Ferrin Katz, Matt Jungkeit, Fall 2019
//
// Sequencer Generic @ 2x Rate for 1Hz Capture
//
// The purpose of this code is to provide an example for how to best
// sequence a set of periodic services for problems similar to and including
// the final project in real-time systems.
//
// For example: Service_1 for camera frame aquisition
//              Service_2 for image analysis and timestamping
//              Service_3 for image processing (difference images)
//              Service_4 for save time-stamped image to file service
//              Service_5 for write difference factor to syslog
//
// At least two of the services need to be real-time and need to run on a single
// core or run without affinity on the SMP cores available to the Linux 
// scheduler as a group.  All services can be real-time, but you could choose
// to make just the first 2 real-time and the others best effort.
//
// For the standard project, to time-stamp images at the 1 Hz rate with unique
// clock images (unique second hand / seconds) per image, you might use the 
// following rates for each service:
//
// Sequencer - 60 Hz, core # 8
//                   [gives semaphores to all other services]
// Service_1 - 3 Hz, every other Sequencer loop, core # 7
//                   [buffers 3 images per second]
// Service_2 - 1 Hz, every 6th Sequencer loop, core # 6
//                   [time-stamp middle sample image with cvPutText or header]
// Service_3 -  
//                   [difference current and previous time stamped images]
// Service_4 - 
//                   [save time stamped image with cvSaveImage or write()]
// With the above, priorities by RM policy would be:
//
// Sequencer = RT_MAX	@ 60 Hz
// Servcie_1 = RT_MAX-1	@ 30 Hz
// Service_2 = RT_MAX-2	@ 10 Hz
// Service_3 = RT_MAX-3	@ 5  Hz
// Service_4 = RT_MAX-2	@ 10 Hz
// Service_5 = RT_MAX-3	@ 5  Hz
//

// This is necessary for CPU affinity macros in Linux
#define _GNU_SOURCE

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>
#include <sys/sysinfo.h>

#include <errno.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef struct
{
    int threadIdx;
    unsigned long long sequencePeriods;
} threadParams_t;

typedef struct queueItem
{
    Mat userInputItem;
    struct queueItem *nextItem;
} QUEUE_ITEM;

#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_SEC (1000000000)
#define TRUE (1)
#define FALSE (0)

#define NUM_THREADS (6+1)

VideoCapture capture;
Mat frame;

int abortTest=FALSE;
int abortS1=FALSE, abortS2=FALSE, abortS3=FALSE, abortS4=FALSE, abortS5=FALSE;
sem_t semS1, semS2, semS3, semS4, semS5;
bool Start = false;
struct timeval start_time_val;

int service1Freq = 12, service2Freq=60, service3Freq=6;


QUEUE_ITEM* rawFramePointer = NULL;
QUEUE_ITEM* outputBuffer = NULL;
QUEUE_ITEM* differenceBuffer = NULL;

Mat frameBuffer[60];
int frameCounter = 0;

/* Define Functions */
void push(Mat data, QUEUE_ITEM **listPointer);
Mat pop(QUEUE_ITEM **listPointer);
int isEmpty(QUEUE_ITEM **listPointer);

void *Sequencer(void *threadp);

void *Service_1(void *threadp);
void *Service_2(void *threadp);
void *Service_3(void *threadp);
void *Service_4(void *threadp);
void *Service_5(void *threadp);
double deltaTime(struct timeval prev_time_val);
char* timeStamp(void);
double getTimeMsec(void);
void print_scheduler(void);


int main(void)
{
    struct timeval current_time_val;
    int i, rc, scope;
    cpu_set_t threadcpu;
    pthread_t threads[NUM_THREADS];
    threadParams_t threadParams[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio;
    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    cpu_set_t allcpuset;

    if(!capture.open(0)) 
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    else
    {
	   std::cout << "Opened default camera interface" << std::endl;
    }

        while(!capture.read(frame))
        {
        	std::cout << "No frame" << std::endl;
        	cv::waitKey();
        }

    printf("Starting High Rate Sequencer Demo\n");
    gettimeofday(&start_time_val, (struct timezone *)0);
    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "START High Rate Sequencer @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

   printf("System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

   CPU_ZERO(&allcpuset);

   for(i=0; i < 8; i++)
       CPU_SET(i, &allcpuset);

   printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));


    // initialize the sequencer semaphores
    //
    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (-1); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S3 semaphore\n"); exit (-1); }
    if (sem_init (&semS4, 0, 0)) { printf ("Failed to initialize S4 semaphore\n"); exit (-1); }
    if (sem_init (&semS5, 0, 0)) { printf ("Failed to initialize S5 semaphore\n"); exit (-1); }

    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();


    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

    for(i=0; i < NUM_THREADS; i++)
    {

      CPU_ZERO(&threadcpu);
      CPU_SET(3, &threadcpu);

      rc=pthread_attr_init(&rt_sched_attr[i]);
      rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
      rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
      //rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

      rt_param[i].sched_priority=rt_max_prio-i;
      pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

      threadParams[i].threadIdx=i;
    }
   
    printf("Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

    // Create Service threads which will block awaiting release for:
    //

    // Servcie_1 = RT_MAX-1	@ 30 Hz
    //
    rt_param[1].sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&rt_sched_attr[1], &rt_param[1]);
    rc=pthread_create(&threads[1],               // pointer to thread descriptor
                      &rt_sched_attr[1],         // use specific attributes
                      //(void *)0,               // default attributes
                      Service_1,                 // thread function entry point
                      (void *)&(threadParams[1]) // parameters to pass in
                     );
    if(rc < 0)
        perror("pthread_create for service 1");
    else
        printf("pthread_create successful for service 1\n");


    // Service_2 = RT_MAX-2	@ 10 Hz
    //
    rt_param[2].sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rc=pthread_create(&threads[2], &rt_sched_attr[2], Service_2, (void *)&(threadParams[2]));
    if(rc < 0)
        perror("pthread_create for service 2");
    else
        printf("pthread_create successful for service 2\n");


    // Service_3 = RT_MAX-3	@ 5 Hz
    //
    rt_param[3].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[3], &rt_param[3]);
    rc=pthread_create(&threads[3], &rt_sched_attr[3], Service_3, (void *)&(threadParams[3]));
    if(rc < 0)
        perror("pthread_create for service 3");
    else
        printf("pthread_create successful for service 3\n");


    // Service_4 = RT_MAX-2	@ 10 Hz
    //
    rt_param[4].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[4], &rt_param[4]);
    rc=pthread_create(&threads[4], &rt_sched_attr[4], Service_4, (void *)&(threadParams[4]));
    if(rc < 0)
        perror("pthread_create for service 4");
    else
        printf("pthread_create successful for service 4\n");


    // Service_5 = RT_MAX-3	@ 5 Hz
    //
    rt_param[5].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[5], &rt_param[5]);
    rc=pthread_create(&threads[5], &rt_sched_attr[5], Service_5, (void *)&(threadParams[5]));
    if(rc < 0)
        perror("pthread_create for service 5");
    else
        printf("pthread_create successful for service 5\n");

    // Wait for service threads to initialize and await relese by sequencer.
    //
    // Note that the sleep is not necessary of RT service threads are created wtih 
    // correct POSIX SCHED_FIFO priorities compared to non-RT priority of this main
    // program.
    //
    // usleep(1000000);
 
    // Create Sequencer thread, which like a cyclic executive, is highest prio
    printf("Start sequencer\n");
    threadParams[0].sequencePeriods=60*2000;

    // Sequencer = RT_MAX	@ 60 Hz
    //
    rt_param[0].sched_priority=rt_max_prio;
    pthread_attr_setschedparam(&rt_sched_attr[0], &rt_param[0]);
    rc=pthread_create(&threads[0], &rt_sched_attr[0], Sequencer, (void *)&(threadParams[0]));
    if(rc < 0)
        perror("pthread_create for sequencer service 0");
    else
        printf("pthread_create successful for sequeencer service 0\n");


   for(i=0;i<NUM_THREADS;i++)
       pthread_join(threads[i], NULL);

   printf("\nTEST COMPLETE\n");
}


void *Sequencer(void *threadp)
{
    struct timeval current_time_val;
    struct timespec delay_time = {0,16666666}; // delay for 16.67 msec, 60 Hz
    struct timespec remaining_time;
    double current_time;
    double residual;
    int rc, delay_cnt=0;
    unsigned long long seqCnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    do
    {
        delay_cnt=0; residual=0.0;

        gettimeofday(&current_time_val, (struct timezone *)0);
        syslog(LOG_CRIT, "Sequencer thread prior to delay @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        do
        {
            rc=nanosleep(&delay_time, &remaining_time);

            if(rc == EINTR)
            { 
                residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

                if(residual > 0.0) printf("residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);
 
                delay_cnt++;
            }
            else if(rc < 0)
            {
                perror("Sequencer nanosleep");
                exit(-1);
            }
           
        } while((residual > 0.0) && (delay_cnt < 100));

        seqCnt++;
        gettimeofday(&current_time_val, (struct timezone *)0);
        //syslog(LOG_CRIT, "Sequencer cycle %llu @ sec=%d, msec=%d\n", seqCnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);


        if(delay_cnt > 1) printf("Sequencer looping delay %d\n", delay_cnt);


        // Release each service at a sub-rate of the generic sequencer rate

        // Servcie_1 = RT_MAX-1	@ 3 Hz
        if((seqCnt % service1Freq) == 0) sem_post(&semS1);

        // Service_2 = RT_MAX-2	@ 1 Hz
        if((seqCnt % service2Freq) == 0) sem_post(&semS2);

        // Service_3 = RT_MAX-3	@ 20 Hz
        if((seqCnt % service3Freq) == 0) sem_post(&semS3);

        // Service_4 = RT_MAX-2	
        if((seqCnt % 30) == 0) sem_post(&semS4);

        // Service_5 = RT_MAX-3	
        if((seqCnt % 10) == 0) sem_post(&semS5);

        //gettimeofday(&current_time_val, (struct timezone *)0);
        //syslog(LOG_CRIT, "Sequencer release all sub-services @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    } while(!abortTest && (seqCnt < threadParams->sequencePeriods));
    sem_post(&semS1); sem_post(&semS2); sem_post(&semS3);
    sem_post(&semS4); sem_post(&semS5);
    abortS1=TRUE; abortS2=TRUE; abortS3=TRUE;
    abortS4=TRUE; abortS5=TRUE;
    cout << "Sequencer Finished" << endl;

    pthread_exit((void *)0);
}



void *Service_1(void *threadp)
{
    struct timeval current_time_val, temp_time_val, startService_time_val;
    double current_time;
    unsigned long long S1Cnt=0;
    char imgText[100];
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Frame Sampler thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Frame Sampler thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS1)
    {
        sem_wait(&semS1);
        gettimeofday(&startService_time_val, (struct timezone *)0);

        while(!capture.read(frame))
        {
        	std::cout << "No frame" << std::endl;
        	cv::waitKey();
        }
        gettimeofday(&current_time_val, (struct timezone *)0);
        current_time_val.tv_sec = current_time_val.tv_sec - 1;
        cvtColor(frame, frame, CV_BGR2GRAY);
        strftime (imgText, sizeof (imgText), "%Y-%m-%d %H:%M:%S", localtime(&current_time_val.tv_sec));
        putText(frame, imgText, cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, false);

        push(frame,&rawFramePointer);
        if(Start)
        {
            frameBuffer[frameCounter%60] = frame;
            frameCounter++;
            syslog(LOG_CRIT, "Frame Sampler deltaT5, %f, %f", deltaTime(current_time_val), deltaTime(start_time_val));
        }

        S1Cnt++;    

        gettimeofday(&current_time_val, (struct timezone *)0);
        syslog(LOG_CRIT, "Frame Sampler WCET, release,%llu,%d\n", S1Cnt, deltaTime(startService_time_val));
        syslog(LOG_CRIT, "Frame Sampler release %llu @ sec=%d, msec=%d\n", S1Cnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    }

    pthread_exit((void *)0);
}


void *Service_2(void *threadp)
{
    struct timeval current_time_val;
    double current_time;
    unsigned long long S2Cnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    char filePath[100];


    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Image Analysis thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Image Analysis thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS2)
    {
        sem_wait(&semS2);
        if(Start)
        {
            sprintf(filePath, "images/img-%d.jpg", S2Cnt);
            //putText(frameBuffer[1], timeStamp(), cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, false); 
            //imwrite(filePath,frameBuffer[1]); 

            S2Cnt++;
            gettimeofday(&current_time_val, (struct timezone *)0);
            //syslog(LOG_CRIT, "Image Analysis release %llu @ sec=%d, msec=%d\n", S2Cnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        }
    }

    pthread_exit((void *)0);
}

void *Service_3(void *threadp)
{
    struct timeval current_time_val, startService_time_val;
    double current_time;
    unsigned long long S3Cnt=0, cntAtLastTick = 0;
    double diffsum, diffPercent;
    unsigned int maxdiff;
    bool firstTime = true, newTick = true;
    char imgText[100], filePath[100];
    Mat tempFrameBuffer[9];
    Rect crop(0, 40, 640, 420);
    Mat rawMat, prevMat, mat_diff, concatMat, satMat;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Difference Image Proc thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Difference Image Proc thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS3)
    {
        sem_wait(&semS3);
        gettimeofday(&startService_time_val, (struct timezone *)0);

        if(!isEmpty(&rawFramePointer) && !firstTime)
        {
            rawMat = pop(&rawFramePointer)(crop);
         	absdiff(rawMat, prevMat, mat_diff);
            maxdiff = (mat_diff.cols)*(mat_diff.rows)*255;
            threshold(mat_diff, mat_diff, 75, 0, CV_THRESH_TOZERO);
            //Detect if image has changed
            diffsum = sum(mat_diff)[0];
            diffPercent = diffsum/maxdiff;
            if (diffPercent > 0.0005)
            {   
                Start = true;
            }

            mat_diff.convertTo(satMat, -1, 10.0, 10.0);
            putText(satMat, timeStamp(), cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, false);  
            sprintf(imgText, "Difference: %f", diffPercent);
            putText(satMat, imgText, cvPoint(30,50), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, false);  
            if (!Start)
                cout << diffPercent << "   " << timeStamp() << endl;
            vconcat(satMat,rawMat,concatMat);
            push(concatMat, &differenceBuffer);
            if(!newTick && diffPercent == 0)
            {
                newTick = true;
                //cout << "newTick" << endl;
            }

            if(diffPercent > 0.001 && !newTick && S3Cnt-cntAtLastTick == 5)
            {
                //One second has passed, but the difference counter hasn't been reset, probably because movement in the background
                newTick = true;
            }    
//Find middle difference image
            if(diffPercent > 0.00001 && frameCounter > 6 && newTick)
            {
                int bufferDiffDum, tickFrame1, tickFrame2;
                double bufferDiffPercent, maxDiffPercent1 = 0, maxDiffPercent2 = 0;

                newTick = false;
                for(int i = 0; i < 7; i++) //Save previous 7 images from the loop buffer
                {
                    tempFrameBuffer[i] = frameBuffer[(frameCounter-i)%60];
                }
                //Search for the previous tick frame
                for(int i = 0; i < 4; i++)
                {
                    //Find difference of all images
                    bufferDiffDum = sum(tempFrameBuffer[i])[0];
                    bufferDiffPercent = diffsum/maxdiff;
                    if(bufferDiffPercent > maxDiffPercent1)
                    {
                        maxDiffPercent1 = bufferDiffPercent;
                        tickFrame1 = i;
                    }
                }
                for(int i = 3; i < 7; i++)
                {
                    //Find difference of all images
                    bufferDiffDum = sum(tempFrameBuffer[i])[0];
                    bufferDiffPercent = diffsum/maxdiff;
                    if(bufferDiffPercent > maxDiffPercent2)
                    {
                        maxDiffPercent1 = bufferDiffPercent;
                        tickFrame2 = i;
                    }
                }
                cntAtLastTick = S3Cnt;
                int middleFrame = (tickFrame1+tickFrame2)/2; //Truncation is needed, don't change type
                push(tempFrameBuffer[middleFrame], &outputBuffer);
            }

            prevMat = rawMat;
            S3Cnt++;
            gettimeofday(&current_time_val, (struct timezone *)0);
            //syslog(LOG_CRIT, "Difference Image Proc release %llu @ sec=%d, msec=%d\n", S3Cnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        }
        else if(!isEmpty(&rawFramePointer))
        {
            rawMat = pop(&rawFramePointer)(crop);
            prevMat = rawMat;
            firstTime = false;
        }
        syslog(LOG_CRIT, "Difference Image Proc WCET, release,%llu,%d\n", S3Cnt, deltaTime(startService_time_val));
    }

    pthread_exit((void *)0);
}

void *Service_4(void *threadp)
{
    struct timeval current_time_val, startService_time_val;
    double current_time;
    unsigned long long S4Cnt=0;
    char filePath[100];
    Mat saveMat;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Tick Image Save to File thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Tick Image Save to File thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS4)
    {
        sem_wait(&semS4);
        gettimeofday(&startService_time_val, (struct timezone *)0);

        if(!isEmpty(&outputBuffer))
        {
            saveMat = pop(&outputBuffer);  
            sprintf(filePath, "images/img-%d.jpg", S4Cnt);
            imwrite(filePath, saveMat);
            S4Cnt++;
            gettimeofday(&current_time_val, (struct timezone *)0);
            syslog(LOG_CRIT, "Tick Image Save WCET, release,%llu,%d\n", S4Cnt, deltaTime(startService_time_val));
            //syslog(LOG_CRIT, "Tick Image Save to File release %llu @ sec=%d, msec=%d\n", S4Cnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        }
    }

    pthread_exit((void *)0);
}

void *Service_5(void *threadp)
{
    struct timeval current_time_val, startService_time_val;
    double current_time;
    unsigned long long S5Cnt=0;
    char filePath[100];
    Mat saveMat;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Processed Image Save to File thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Difference Image Save to File thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS5)
    {
        sem_wait(&semS5);
        gettimeofday(&startService_time_val, (struct timezone *)0);
        
        
        if(!isEmpty(&differenceBuffer))
        {
            saveMat = pop(&differenceBuffer);  
            sprintf(filePath, "differences/img-%d.jpg", S5Cnt);
            imwrite(filePath, saveMat);

            S5Cnt++;
            gettimeofday(&current_time_val, (struct timezone *)0);
            syslog(LOG_CRIT, "Difference Image Save WCET, release,%llu,%d\n", S5Cnt, deltaTime(startService_time_val));
        }

    }

    pthread_exit((void *)0);
}


//****************************************************************************//

double deltaTime(struct timeval prev_time_val)
{
    struct timeval current_time_val;
    double deltaT;
    gettimeofday(&current_time_val, (struct timezone *)0);
    current_time_val.tv_sec--;
    deltaT = (current_time_val.tv_sec-prev_time_val.tv_sec) + (((float)current_time_val.tv_usec-prev_time_val.tv_usec)/1000000);
    
    return deltaT;
}
char* timeStamp(void)
{
    struct timeval current_time_val;
    static char imgText[100];
    
    gettimeofday(&current_time_val, (struct timezone *)0);
    current_time_val.tv_sec = current_time_val.tv_sec - 1;
    strftime (imgText, sizeof (imgText), "%Y-%m-%d %H:%M:%S", localtime(&current_time_val.tv_sec));
    //cout << imgText << endl;
    return imgText;
}

double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};

  clock_gettime(CLOCK_MONOTONIC, &event_ts);
  return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}


void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n"); exit(-1);
           break;
       default:
           printf("Pthread Policy is UNKNOWN\n"); exit(-1);
   }
}

void push(Mat data, QUEUE_ITEM **listPointer)
{
    QUEUE_ITEM* insertionPointer;
    insertionPointer = new QUEUE_ITEM;
    insertionPointer->userInputItem = data;
    if(*listPointer == NULL)
    {
        insertionPointer->nextItem = insertionPointer;
        *listPointer = insertionPointer;
    }
    else if((*listPointer)->nextItem == (*listPointer))
    {
        insertionPointer->nextItem = *listPointer;
        (*listPointer)->nextItem = insertionPointer;
        *listPointer = (*listPointer)->nextItem;
    }
    else
    {
        insertionPointer->nextItem = (*listPointer)->nextItem;
        (*listPointer)->nextItem = insertionPointer;
        *listPointer = (*listPointer)->nextItem;
    }

}

Mat pop(QUEUE_ITEM **listPointer)
{
    Mat toReturn;
    QUEUE_ITEM* deletionPointer;

    deletionPointer = (*listPointer)->nextItem;
    toReturn = deletionPointer->userInputItem;

    if((*listPointer)->nextItem != (*listPointer))
    {
        (*listPointer)->nextItem = deletionPointer->nextItem;
    }
    else
    {
        *listPointer = NULL;
    }

    delete deletionPointer;

    return toReturn;
}

int isEmpty(QUEUE_ITEM **listPointer)
{
    if(*listPointer == NULL)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
