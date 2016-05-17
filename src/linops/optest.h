
typedef void (*fun_t)(void*, complex float*, const complex float*);


extern _Bool test_adjoint_fun(long N, long M, void* data, fun_t frw, fun_t adj, _Bool gpu);
extern _Bool test_derivative_fun(long N, long M, void* data, fun_t frw, fun_t der);

struct operator_s;
extern _Bool test_adjoint_op(const struct operator_s* frw, const struct operator_s* adj, _Bool gpu);

struct linop_s;
extern _Bool test_adjoint_linop(const struct linop_s* linop, _Bool gpu);
