/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 *
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
 
#include "mm.h"
#include "memlib.h"
 
/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "1190300321",
    /* First member's full name */
    "Zheng Shenghe",
    /* First member's email address */
    "531905990@qq.com",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""
};
/* Basic constants and macros */

#define WSIZE     4
#define DSIZE     8
#define INITCHUNKSIZE (1<<6)
#define CHUNKSIZE (1<<12)
#define MAX_LEN     16
#define ALIGN(size) ((((size) + (DSIZE-1)) / (DSIZE)) * (DSIZE)) //����

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
 
#define PACK(size, alloc) ((size) | (alloc))
 
/* Read and write a word at address p */
#define GET(p)       (*(size_t *)(p))
#define PUT(p, val)  (*(size_t *)(p) = (val))
 
/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)
 
/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp)       ((char *)(bp) - WSIZE)
#define FTRP(bp)       ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)
 
/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)  ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp)  ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))
 
#define SET_PTR(p, bp) (*(unsigned int *)(p) = (unsigned int)(bp))
 
#define PRED_PTR(bp) ((char *)(bp))
#define SUCC_PTR(bp) ((char *)(bp) + WSIZE)
 
#define PRED(bp) (*(char **)(bp))
#define SUCC(bp) (*(char **)(SUCC_PTR(bp)))
 
/* Global variables */
static char *heap_listp;  /* pointer to first block */ 
static void *extend_heap(size_t size);
static void *coalesce(void *bp);
static void *place(void *bp, size_t size);
static void printblock(void *bp);
static void checkblock(void *bp);
static void InsertNode(void *bp, size_t size); //���뵽��������
static void DeleteNode(void *bp);  //ɾ��
void *Lists[MAX_LEN];  //����������� 
 
/*��ʼ���ڴ������*/
int mm_init(void)
{
    int i;
    /* ��ʼ������������� */
    for (i = 0; i < MAX_LEN; i++)
    {
        Lists[i] = NULL;
    }
    if ((heap_listp = mem_sbrk(4*WSIZE)) == NULL)
        return -1;
    PUT(heap_listp, 0);   //�������
    PUT(heap_listp + (1 * WSIZE), PACK(DSIZE, 1));  //���Կ�
    PUT(heap_listp + (2 * WSIZE), PACK(DSIZE, 1));
    PUT(heap_listp + (3 * WSIZE), PACK(0, 1));  //��β��
 
    /* Extend the empty heap with a free block of INITCHUNKSIZE bytes */
    if (extend_heap(INITCHUNKSIZE) == NULL)
    return -1;
    return 0;
}
 
/*mm_free - Free a block*/
void mm_free(void *bp)
{
    size_t size = GET_SIZE(HDRP(bp));
 
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    InsertNode(bp, size);
    coalesce(bp);
}
 
/*��չ��*/
static void *extend_heap(size_t size)
{
    char *bp;
    /* Allocate an even number of words to maintain alignment */
    size = ALIGN(size);
    if ((bp = mem_sbrk(size)) == (void *)-1)
        return NULL;
    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));
    /*���뵽������б��� */
    InsertNode(bp, size);
    /* Coalesce if the previous block was free */
    return coalesce(bp);
}
 
/*���뵽��������*/
static void InsertNode(void *bp, size_t size)
{
    int i = 0;
    void *search_bp = NULL;
    void *insert_bp = NULL;
 
    while((i<MAX_LEN-1)&&(size>1))  // ����size�Ĵ�С�ҵ���Ӧ�ķ����������
    {
        size >>= 1;
        i++;
    }
    /* �ҵ�������������ڸ�����Ѱ�Ҷ�Ӧ�Ĳ���λ�ã����ұ������п���С����ֲ� */
    search_bp = Lists[i];
    while ((search_bp != NULL) && (size > GET_SIZE(HDRP(search_bp))))
    {
        insert_bp = search_bp;
        search_bp = PRED(search_bp);
    }
 
    /* ������� */
    if (search_bp != NULL)
    {
        /* ���м����*/
        if (insert_bp != NULL)
        {
            SET_PTR(PRED_PTR(bp), search_bp);
            SET_PTR(SUCC_PTR(search_bp), bp);
            SET_PTR(SUCC_PTR(bp), insert_bp);
            SET_PTR(PRED_PTR(insert_bp), bp);
        }
        /* 2. �ڿ�ͷ����*/
        else
        {
            SET_PTR(PRED_PTR(bp), search_bp);
            SET_PTR(SUCC_PTR(search_bp), bp);
            SET_PTR(SUCC_PTR(bp), NULL);
            Lists[i] = bp;
        }
    }
    else
    {
        if (insert_bp != NULL)
        { /*�ڽ�β����*/
            SET_PTR(PRED_PTR(bp), NULL);
            SET_PTR(SUCC_PTR(bp), insert_bp);
            SET_PTR(PRED_PTR(insert_bp), bp);
        }
        else
        { /*��һ�β��� */
            SET_PTR(PRED_PTR(bp), NULL);
            SET_PTR(SUCC_PTR(bp), NULL);
            Lists[i] = bp;
        }
    }
}
 
/*�ӿ���������ɾ��*/
static void DeleteNode(void *bp)
{
    int i = 0;
    size_t size = GET_SIZE(HDRP(bp));
 
    // ����size�Ĵ�С�ҵ���Ӧ�ķ����������
    while ((i < MAX_LEN - 1) && (size > 1))
    {
        size >>= 1;
        i++;
    }
    /* ���ֿ����� */
    if (PRED(bp) != NULL)
    {
        /* �м�ɾ�� */
        if (SUCC(bp) != NULL)
        {
            SET_PTR(SUCC_PTR(PRED(bp)), SUCC(bp));
            SET_PTR(PRED_PTR(SUCC(bp)), PRED(bp));
        }
        /* ��ͷɾ���������п�*/
        else
        {
            SET_PTR(SUCC_PTR(PRED(bp)), NULL);
            Lists[i] = PRED(bp);
        }
    }
    else
    {
        /* 3. ��βɾ��*/
        if (SUCC(bp) != NULL)
        {
            SET_PTR(PRED_PTR(SUCC(bp)), NULL);
        }
        /* 4. ��һ��ɾ��*/
        else
        {
            Lists[i] = NULL;
        }
    }
}
 
/*�ϲ���*/
static void *coalesce(void *bp)
{
    size_t  prev_alloc = GET_ALLOC(HDRP(PREV_BLKP(bp)));
    size_t  next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t size = GET_SIZE(HDRP(bp));
    /*�������*/
    if (prev_alloc && next_alloc)   /*case1*/
    {
        return bp;
    }
 
    else if (prev_alloc && !next_alloc)   /*case2*/
    {
        DeleteNode(bp);
        DeleteNode(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
    }
 
    else if (!prev_alloc && next_alloc)   /*case3*/
    {
        DeleteNode(bp);
        DeleteNode(PREV_BLKP(bp));
        size += GET_SIZE(HDRP(PREV_BLKP(bp)));
        PUT(FTRP(bp), PACK(size, 0));
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
    }
 
    else   /*case4*/
    {
        DeleteNode(bp);
        DeleteNode(PREV_BLKP(bp));
        DeleteNode(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(HDRP(NEXT_BLKP(bp)));
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
    }
 /* �ϲ����free����뵽�������ӱ��� */
    InsertNode(bp, size);
 
    return bp;
}
 
/*�����*/
void *mm_malloc(size_t size)
{
    char *bp = NULL;
    int i = 0;
    if (size == 0)
        return NULL;
 
    if (size <= DSIZE)
        size = 2*DSIZE;
    else
        size = ALIGN(size+DSIZE);  //�ڴ����
 
    size_t asize = size;
 
    while (i < MAX_LEN)
    {
        /* ���Һ��ʵĿ������� */
        if (((asize <= 1) && (Lists[i] != NULL)))
        {
            bp = Lists[i];
            /* �ҵ������ڸ���Ѱ�Ҵ�С���ʵ�δ����� */
            while ((bp != NULL) && ((size > GET_SIZE(HDRP(bp)))))
                bp = PRED(bp);
 
            /* �ҵ���Ӧ��δ����Ŀ� */
            if (bp != NULL)
                break;
        }
        asize >>= 1;
        i++;
    }
 
    /* û���ҵ����ʵ�δ����飬����չ�� */
    if (bp == NULL){
        if ((bp = extend_heap(MAX(size, CHUNKSIZE))) == NULL)
            return NULL;
    }
    /* ��δ�������allocate size��С�Ŀ� */
    bp = place(bp, size);
 
    return bp;
}
 
/*����С�ֽڵĿ���ڿ��п�bp�Ŀ�ʼ�����������������������С���С����*/
static void *place(void *bp, size_t asize)
{
    size_t csize = GET_SIZE(HDRP(bp));
    size_t remaining = csize - asize; /* allocate size��С�Ŀռ��ʣ��Ĵ�С */
 
    DeleteNode(bp);
 
    /* ���ʣ��Ĵ�СС����С�飬�򲻷���ԭ�� */
    if (remaining < DSIZE * 2)
    {
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
 
    else if (asize >= 96)
    {
        PUT(HDRP(bp), PACK(remaining, 0));
        PUT(FTRP(bp), PACK(remaining, 0));
        PUT(HDRP(NEXT_BLKP(bp)), PACK(asize, 1));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(asize, 1));
        InsertNode(bp, remaining);
        return NEXT_BLKP(bp);
    }
 
    else
    {
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        PUT(HDRP(NEXT_BLKP(bp)), PACK(remaining, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(remaining, 0));
        InsertNode(NEXT_BLKP(bp), remaining);
    }
    return bp;
}
 
/*�Ľ�������·��亯��*/
void *mm_realloc(void *bp, size_t size)
{
    void *new_p = bp;
    int remaining;
    /*Ingore spurious requests*/
    if (size == 0)
        return NULL;
 
    if (size <= DSIZE)
        size = 2 * DSIZE;
    else
        size = ALIGN(size + DSIZE);  //�ڴ����
 
    /* ���sizeС��ԭ����Ĵ�С��ֱ�ӷ���ԭ���Ŀ� */
    if ((remaining = GET_SIZE(HDRP(bp)) - size) >= 0)
        return bp;
 
    /* �����ȼ���ַ������һ�����Ƿ�Ϊδ�������߸ÿ��ǶѵĽ����� */
    else if (!GET_ALLOC(HDRP(NEXT_BLKP(bp))) || !GET_SIZE(HDRP(NEXT_BLKP(bp))))
    {
        /* ������Ϻ���������ַ�ϵ�δ�����ռ�Ҳ��������ô��Ҫ��չ�� */
        if ((remaining =GET_SIZE(HDRP(bp))+GET_SIZE(HDRP(NEXT_BLKP(bp)))-size)<0)
        {
            if (extend_heap(MAX(-remaining, CHUNKSIZE)) == NULL)
                return NULL;
            remaining +=MAX(-remaining,CHUNKSIZE);
        }
 
        /* �ӷ������������ɾ���ո����õ�δ����鲢�����¿��ͷβ */
        DeleteNode(NEXT_BLKP(bp));
        PUT(HDRP(bp), PACK(size + remaining, 1));
        PUT(FTRP(bp), PACK(size + remaining, 1));
    }
    /* ���û�п������õ�����δ����飬ֻ�������µĲ�������δ����� */
    else
    {
        new_p = mm_malloc(size);
        memcpy(new_p, bp, GET_SIZE(HDRP(bp)));
        mm_free(bp);
    }
    return new_p;
}
 
/*���ѵ�һ����*/
void mm_checkheap(int verbose)
{
    char *bp = heap_listp;
 
    if (verbose)
	printf("Heap (%p):\n", heap_listp);
 
    if ((GET_SIZE(HDRP(heap_listp)) != DSIZE) || !GET_ALLOC(HDRP(heap_listp)))
	printf("Bad prologue header\n");
    checkblock(heap_listp);
 
    for (bp = heap_listp; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
	if (verbose)
	    printblock(bp);
	checkblock(bp);
    }
 
    if (verbose)
	printblock(bp);
    if ((GET_SIZE(HDRP(bp)) != 0) || !(GET_ALLOC(HDRP(bp))))
	printf("Bad epilogue header\n");
}
 
static void printblock(void *bp)
{
    size_t hsize, halloc, fsize, falloc;
 
    hsize = GET_SIZE(HDRP(bp));
    halloc = GET_ALLOC(HDRP(bp));
    fsize = GET_SIZE(FTRP(bp));
    falloc = GET_ALLOC(FTRP(bp));
 
    if (hsize == 0) {
	printf("%p: EOL\n", bp);
	return;
    }
 
    printf("%p: header: [%d:%c] footer: [%d:%c]\n", bp,
	   hsize, (halloc ? 'a' : 'f'),
	   fsize, (falloc ? 'a' : 'f'));
}
 
static void checkblock(void *bp)
{
    if ((size_t)bp % 8)
	printf("Error: %p is not doubleword aligned\n", bp);
    if (GET(HDRP(bp)) != GET(FTRP(bp)))
	printf("Error: header does not match footer\n");
}