#include <stdio.h>

#include "libeasynn.h"
#include "program.h"
#include "evaluation.h"

program *create_program()
{
    program *prog = new program;
    return prog;
}

void append_expression(
    program *prog,
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    prog->append_expression(expr_id, op_name, op_type, inputs, num_inputs);
}

int add_op_param_double(
    program *prog,
    const char *key,
    double value)
{
    return prog->add_op_param_double(key, value);
}

int add_op_param_ndarray(
    program *prog,
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    return prog->add_op_param_ndarray(key, dim, shape, data);
}

evaluation *build(program *prog)
{
    evaluation *eval = prog->build();
    return eval;
}

void add_kwargs_double(
    evaluation *eval,
    const char *key,
    double value)
{
    eval->add_kwargs_double(key, value);
}

void add_kwargs_ndarray(
    evaluation *eval,
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    eval->add_kwargs_ndarray(key, dim, shape, data);
}

int execute(
    evaluation *eval,
    int *p_dim,
    size_t **p_shape,
    double **p_data)
{
    int ret = eval->execute();
    tensor* res = eval->get_result();
    *p_dim = res->get_dim();
    *p_shape = res->get_shape_array();
    *p_data = res->get_data_array();
    fflush(stdout);
    return 0;
}
