#include "futcache/box.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct { double *lo; double *hi; } box_rect_t;
static void *box_default_allocate(void *context, size_t size) { (void)context; return malloc(size); }
static void box_default_deallocate(void *context, void *pointer) { (void)context; free(pointer); }
static futcache_status_t box_normalize_allocator(const futcache_allocator_t *requested, futcache_allocator_t *normalized) {
    if (requested == NULL || (requested->allocate == NULL && requested->deallocate == NULL)) {
        normalized->allocate = box_default_allocate;
        normalized->deallocate = box_default_deallocate;
        normalized->context = NULL;
        return FUTCACHE_OK;
    }
    if (requested->allocate == NULL || requested->deallocate == NULL) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *normalized = *requested;
    return FUTCACHE_OK;
}
struct futcache_box {
    futcache_box_config_t c;
    futcache_allocator_t allocator;
    double *min; double *max;
    box_rect_t *rects;
    size_t count, capacity, memory;
    uint64_t observations, novel, generation;
    size_t peak_count;
    pthread_rwlock_t lock;
};
static void *a_malloc(const futcache_allocator_t *a, size_t n) { return a->allocate(a->context, n); }
static void a_free(const futcache_allocator_t *a, void *p) { if (p != NULL) a->deallocate(a->context, p); }
static bool valid_point(const futcache_box_t *x, const double *p) {
    if (p == NULL) return false;
    for (size_t i=0; i<x->c.dimension; ++i) if (!isfinite(p[i])) return false;
    return true;
}
static bool in_domain(const futcache_box_t *x, const double *p) {
    for (size_t i=0; i<x->c.dimension; ++i) if (p[i] < x->min[i] || p[i] > x->max[i]) return false;
    return true;
}
static bool covered(const futcache_box_t *x, const double *p) {
    for (size_t j=0; j<x->count; ++j) { bool inside=true; for (size_t i=0; i<x->c.dimension; ++i) if (p[i] < x->rects[j].lo[i] || p[i] > x->rects[j].hi[i]) { inside=false; break; } if (inside) return true; }
    return false;
}
void futcache_box_config_init(futcache_box_config_t *c) { if (c != NULL) { memset(c,0,sizeof(*c)); c->dimension=1; } }
futcache_status_t futcache_box_create(const futcache_box_config_t *c, futcache_box_t **out) {
    if (c==NULL || out==NULL || c->dimension==0 || c->dimension>8 || !isfinite(c->epsilon) || c->epsilon<0.0 || c->domain_min==NULL || c->domain_max==NULL || (c->allocator.allocate == NULL) != (c->allocator.deallocate == NULL)) return FUTCACHE_ERROR_INVALID_ARGUMENT;
    *out = NULL;
    futcache_allocator_t allocator;
    futcache_status_t allocator_status = box_normalize_allocator(&c->allocator, &allocator);
    if (allocator_status != FUTCACHE_OK) return allocator_status;
    futcache_box_t *x=a_malloc(&allocator,sizeof(*x)); if (x==NULL) return FUTCACHE_ERROR_OUT_OF_MEMORY; memset(x,0,sizeof(*x)); x->c=*c; x->allocator=allocator;
    if (pthread_rwlock_init(&x->lock,NULL)!=0) { a_free(&allocator,x); return FUTCACHE_ERROR_SYSTEM; }
    size_t bytes=c->dimension*sizeof(double); x->min=a_malloc(&allocator,bytes); x->max=a_malloc(&allocator,bytes); if (x->min==NULL || x->max==NULL) { a_free(&allocator,x->min); a_free(&allocator,x->max); a_free(&allocator,x); return FUTCACHE_ERROR_OUT_OF_MEMORY; }
    memcpy(x->min,c->domain_min,bytes); memcpy(x->max,c->domain_max,bytes); for(size_t i=0;i<c->dimension;++i) if(!isfinite(x->min[i])||!isfinite(x->max[i])||x->min[i]>=x->max[i]) { futcache_box_destroy(x); return FUTCACHE_ERROR_INVALID_ARGUMENT; }
    x->memory=sizeof(*x)+2*bytes; *out=x; return FUTCACHE_OK;
}
void futcache_box_destroy(futcache_box_t *x) { if(x==NULL)return; for(size_t j=0;j<x->count;++j){a_free(&x->allocator,x->rects[j].lo);a_free(&x->allocator,x->rects[j].hi);} a_free(&x->allocator,x->rects); a_free(&x->allocator,x->min); a_free(&x->allocator,x->max); pthread_rwlock_destroy(&x->lock); a_free(&x->allocator,x); }
futcache_status_t futcache_box_is_novel(const futcache_box_t *x,const double *p,bool *out){if(x==NULL||out==NULL||!valid_point(x,p))return FUTCACHE_ERROR_INVALID_ARGUMENT;if(!in_domain(x,p))return FUTCACHE_ERROR_OUT_OF_RANGE;pthread_rwlock_rdlock((pthread_rwlock_t*)&x->lock);*out=!covered(x,p);pthread_rwlock_unlock((pthread_rwlock_t*)&x->lock);return FUTCACHE_OK;}
futcache_status_t futcache_box_observe(futcache_box_t *x,const double *p,bool *out){if(x==NULL||!valid_point(x,p))return FUTCACHE_ERROR_INVALID_ARGUMENT;if(!in_domain(x,p))return FUTCACHE_ERROR_OUT_OF_RANGE;pthread_rwlock_wrlock(&x->lock);if(covered(x,p)){x->observations++;if(out!=NULL)*out=false;pthread_rwlock_unlock(&x->lock);return FUTCACHE_OK;}size_t d=x->c.dimension, bytes=d*sizeof(double);if(x->count==x->capacity){size_t nc=x->capacity==0?8:x->capacity*2;box_rect_t *nr=a_malloc(&x->allocator,nc*sizeof(*nr));if(nr==NULL){pthread_rwlock_unlock(&x->lock);return FUTCACHE_ERROR_OUT_OF_MEMORY;}if(x->rects!=NULL)memcpy(nr,x->rects,x->count*sizeof(*nr));a_free(&x->allocator,x->rects);x->rects=nr;x->capacity=nc;}double *lo=a_malloc(&x->allocator,bytes),*hi=a_malloc(&x->allocator,bytes);if(lo==NULL||hi==NULL){a_free(&x->allocator,lo);a_free(&x->allocator,hi);pthread_rwlock_unlock(&x->lock);return FUTCACHE_ERROR_OUT_OF_MEMORY;}for(size_t i=0;i<d;++i){lo[i]=fmax(x->min[i],p[i]-x->c.epsilon);hi[i]=fmin(x->max[i],p[i]+x->c.epsilon);}x->rects[x->count++] = (box_rect_t){lo,hi};if(x->count>x->peak_count)x->peak_count=x->count;x->observations++;x->novel++;x->generation++;x->memory+=2*bytes;if(out!=NULL)*out=true;if(x->count>x->capacity)x->capacity=x->count;pthread_rwlock_unlock(&x->lock);return FUTCACHE_OK;}
futcache_status_t futcache_box_get_stats(const futcache_box_t*x,futcache_box_stats_t*out){if(x==NULL||out==NULL)return FUTCACHE_ERROR_INVALID_ARGUMENT;pthread_rwlock_rdlock((pthread_rwlock_t*)&x->lock);out->observations=x->observations;out->novel_observations=x->novel;out->generation=x->generation;out->box_count=x->count;out->peak_box_count=x->peak_count;out->memory_bytes=x->memory;pthread_rwlock_unlock((pthread_rwlock_t*)&x->lock);return FUTCACHE_OK;}
futcache_status_t futcache_box_clear(futcache_box_t*x){if(x==NULL)return FUTCACHE_ERROR_INVALID_ARGUMENT;pthread_rwlock_wrlock(&x->lock);for(size_t j=0;j<x->count;++j){a_free(&x->allocator,x->rects[j].lo);a_free(&x->allocator,x->rects[j].hi);}x->count=0;x->observations=0;x->novel=0;x->peak_count=0;x->memory=sizeof(*x)+2*x->c.dimension*sizeof(double);x->generation++;pthread_rwlock_unlock(&x->lock);return FUTCACHE_OK;}
futcache_status_t futcache_box_validate(const futcache_box_t*x){if(x==NULL)return FUTCACHE_ERROR_INVALID_ARGUMENT;pthread_rwlock_rdlock((pthread_rwlock_t*)&x->lock);for(size_t j=0;j<x->count;++j)for(size_t i=0;i<x->c.dimension;++i)if(x->rects[j].lo[i]<x->min[i]||x->rects[j].hi[i]>x->max[i]||x->rects[j].lo[i]>x->rects[j].hi[i]){pthread_rwlock_unlock((pthread_rwlock_t*)&x->lock);return FUTCACHE_ERROR_SYSTEM;}pthread_rwlock_unlock((pthread_rwlock_t*)&x->lock);return FUTCACHE_OK;}
