import numpy as np
import torch
from os.path import join, dirname
import sys
import time
from traceback import print_exc

def index_select(bitstrings, inds):
    '''
    select bitstrings in specific indices

    :param bitstrings: list of bitstring
    :param inds: indices to be selected
    '''
    return [''.join(bitstring[i] for i in inds) for bitstring in bitstrings]


def combine_bitsting(bitstring_i, bitstring_j, loc_i, loc_j):
    '''
    combine bitstring with two partial ones with their location

    :param bitstring_i: bitstring i to be combined
    :param bitstring_j: bitstring j to be combined
    :param loc_i: location of bitstring i
    :param loc_j: location of bitstring j
    '''
    return ''.join([bitstring_i[loc_i.index(k)] if k in loc_i else bitstring_j[loc_j.index(k)] for k in range(len(loc_i) + len(loc_j))])


def contraction_scheme_multibitstrings_test(eq_sep, order, final_qubits=None, bitstrings=None):
    '''
    construct a scheme by given order with multiple bitstings

    :param eq_sep: list of equations represent the tensor network
    :param order: given order
    :param final_qubits: ids of which tensor is a final qubit
    :param bitstrings: bitstrings will be sampled in the simulation

    :return contraction_scheme: constructed contraction scheme
    :return eq_sep[i]: equation of final tensor
    :return tmp_bitstrings: ordered corresponding bitstrings to amplitudes calculated according to contraction scheme
    '''
    contraction_scheme = []
    if type(final_qubits) == frozenset or type(final_qubits) == set:
        final_qubits = sorted(list(final_qubits))
    tensor_bitstrings = [np.array([0, 1]) for k in range(len(final_qubits))]
    tensor_info = [([], np.array([-1])) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in range(len(eq_sep))]
    
    for edge in order:
        t0 = time.time()
        i, j = edge
        ei, ej = eq_sep[i], eq_sep[j]

        tmp_final_qubit_ids = sorted(tensor_info[i][0] + tensor_info[j][0])
        if len(tmp_final_qubit_ids) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([0])]]
            tmp_bitstrings_rep = np.array([-1])
        elif len(tensor_info[i][0]) > 0  and len(tensor_info[j][0]) == 0:
            batch_sep_sorted = [[torch.tensor([k for k in range(len(tensor_info[i][1]))])], [torch.tensor([0])]]
            tmp_bitstrings_rep = tensor_info[i][1]
        elif len(tensor_info[j][0]) > 0  and len(tensor_info[i][0]) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([k for k in range(len(tensor_info[j][1]))])]]
            tmp_bitstrings_rep = tensor_info[j][1]
        else:
            idx = int(len(tensor_info[i][1]) > len(tensor_info[j][1]))
            tmp_bitstrings = np.unique(index_select(bitstrings, tmp_final_qubit_ids))
            tmp_bitstrings_rep = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
            bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings, [tmp_final_qubit_ids.index(k) for k in tensor_info[i][0]]), index_select(tmp_bitstrings, [tmp_final_qubit_ids.index(k) for k in tensor_info[j][0]])
            bitstrings_rep_i, bitstrings_rep_j = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_i]), np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_j])
            batch_sep = np.array([[np.argwhere(tensor_info[i][1] == b_i)[0][0], np.argwhere(tensor_info[j][1] == b_j)[0][0]] for b_i, b_j in zip(bitstrings_rep_i, bitstrings_rep_j)])
            sort_inds = np.argsort(batch_sep[:, idx])
            batch_sep = batch_sep[sort_inds].T.reshape(2, -1)
            uni, inds = np.unique(batch_sep[idx], return_index=True)
            inds = list(inds) + [len(batch_sep[idx])]
            batch_sep_sorted = [[], []]
            for k in range(len(uni)):
                batch_sep_sorted[idx].append(torch.tensor([uni[k]]))
                batch_sep_sorted[1-idx].append(torch.from_numpy(batch_sep[1-idx][inds[k]:inds[k+1]]))
            tmp_bitstrings_rep = tmp_bitstrings_rep[sort_inds]

        common_indices = sorted(frozenset(ei) & frozenset(ej))

        idxi_j = []
        idxj_i = []
        for idx in common_indices:
            idxi_j.append(ei.index(idx))
            idxj_i.append(ej.index(idx))
        eq_sep[i] = ''.join(
            [ei[m] for m in range(len(ei)) if m not in idxi_j] + [ej[n] for n in range(len(ej)) if n not in idxj_i]
        )
        permute_dim_i, permute_dim_j = 0, 0
        if len(tensor_info[i][0]):
            idxi_j = [ind + 1 for ind in idxi_j]
            permute_dim_i += 1
        if len(tensor_info[j][0]):
            idxj_i = [ind + 1 for ind in idxj_i]
            permute_dim_j += 1
        if permute_dim_j == 1:
            print(permute_dim_i, permute_dim_j, len(ei), len(ej), len(common_indices))
            permute_seq = list(range(permute_dim_i + permute_dim_j + len(ei) + len(ej) - 2 * len(common_indices)))
            si = permute_seq.pop(permute_dim_i + len(ei) - len(common_indices))
            permute_seq.insert(permute_dim_i, si)
            rshape = (-1,) + (2,) * (len(ei) + len(ej) - 2 * len(common_indices))
            next_tensor_shape = (len(tmp_bitstrings_rep),) + (2,) * (len(ei) + len(ej) - 2 * len(common_indices))
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted, permute_seq, rshape, next_tensor_shape))
        else:
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted))
        
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, eq_sep[i], tmp_bitstrings

def tensor_contraction_multibitstrings_test(tensors, contraction_scheme):
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: list of contraction step, defintion of entries in each step:
                               step[0]: locations of tensors to be contracted
                               step[1]: set to be 'tensordot' here, maybe more operations in the future
                               step[2] and step[3], indices arguments of tensordot
                               step[4]: batch dimension of the contraction
                               step[5]: optional, if the second tensor has batch dimension, then here is the permute sequence
                               step[6]: optional, if the second tensor has batch dimension, then here is the reshape sequence

    :return tensors[i]: the final resulting amplitudes
    '''
    for step in contraction_scheme:
        i, j = step[0]
        assert step[1] == 'tensordot'
        if step[1] == 'tensordot':
            batch_i, batch_j = step[4]
            if len(batch_i) > 1:
                tensors[i] = [tensors[i]]
                for k in range(len(batch_i)-1, -1, -1):
                    if k != 0:
                        try:
                            tensors[i].insert(
                                1, 
                                torch.tensordot(
                                    tensors[i][0][batch_i[k]], 
                                    tensors[j][batch_j[k]], 
                                    (step[2], step[3])
                                ).permute(step[5]).reshape(step[6])
                            )
                        except:
                            print(step[0], tensors[i][0][batch_i[k]].shape, tensors[j][batch_j[k]].shape, step[2:])
                            print_exc()
                            sys.exit(1)
                    else:
                        # tensors[i][0] = tensors[i][0][batch_i[k]]
                        try:
                            tensors[i][0] = torch.tensordot(
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                                (step[2], step[3])
                            ).permute(step[5]).reshape(step[6])
                            
                        except:
                            print(step[0], len(batch_i), tensors[i][0].shape, tensors[j].shape, len(step[2]))
                            print_exc()
                            sys.exit(1)
                tensors[j] = []
                try:
                    tensors[i] = torch.cat(tensors[i], dim=0)
                except:
                    print(step, len(tensors[i]))
                    print_exc()
                    sys.exit(1)
            elif len(step) > 5:
                try:
                    tensors[i] = torch.tensordot(
                        tensors[i], 
                        tensors[j], 
                        (step[2], step[3])
                    ).permute(step[5]).reshape(step[6])
                except:
                    print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2:])
                    print_exc()
                    sys.exit(1)
                tensors[j] = []
            else:
                try:
                    # torch.cuda.empty_cache()
                    tensors[i] = torch.tensordot(tensors[i], tensors[j], (step[2], step[3]))
                except:
                    print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2], step[3])
                    print_exc()
                    sys.exit(1)
                tensors[j] = []

    return tensors[i]

def circuit_simulation_new(simulation_data, task_id=0, task_num=0, device='cpu', subtasks_num='all', get_time=False):
    '''
    circuit simultion according to simulation data

    :param simulation_data: dict of all data needed in the simulation, defintion of entries:
                            ['tensors']: numerical tensors corresponding to the tensor network
                            ['slicing_edges_loop']: slicing edges in the interface between head and tail part
                            ['slicing_edges_front']: slicing edges in the head part
                            ['slicing_edges_back']: slicing edges in the tail part
                            ['slicing_indices_dict']: dict of indices for all slicing edges, format of edge: indices
                            ['slicing_edges_companion_dict']: dict of companion edges, see defintion in the main article, format of edge: edge
                            ['scheme']: contraction scheme
                            ['partition']: the head and tail part
                            ['permute_idx']: permute sequence of the final amplitude, in order to make the order the same as their qubit order
                            ['shape_final']: the shape of final amplitude
                            ['shape_inter']: the shape of tensor after contraction of the head part
    :param task_id: number for tag which sequence of sub-tasks will be running
    :param task_num: number to determine how many (2**task_num) sub-tasks will be running
    :param device: device to perform the contraction, format: 'cuda:x' or 'cpu'
    :param get_time: boolean value, if True, return running time of each sub-task

    :return tensors[i]: the final resulting amplitudes
    '''
    tensors = simulation_data['tensors']
    slicing_edges_loop = simulation_data['slicing_edges_loop']
    slicing_edges_front = simulation_data['slicing_edges_front']
    slicing_edges_back = simulation_data['slicing_edges_back']
    slicing_indices_dict = simulation_data['slicing_indices_dict']
    slicing_edges_companion_dict = simulation_data['slicing_edges_companion_dict']
    scheme = simulation_data['scheme']
    partition = simulation_data['partition']
    permute_idx = simulation_data['permute_idx']
    collect_tensor_all = torch.zeros(simulation_data['shape_final'], dtype=torch.complex64, device=device) # None
    if subtasks_num == 'all':
        tasks_front = 2**len(slicing_edges_front)
        tasks_back = 2**len(slicing_edges_back)
    else:
        tasks_front = subtasks_num
        tasks_back = subtasks_num
    
    if get_time:
        torch.cuda.synchronize(device)
        t_start = time.time()
    for s in range(task_id * 2**task_num, (task_id + 1) * 2**task_num):
        # print(f'{s+1}/{(task_id + 1) * 2**task_num}')
        bitstring = list(map(int, np.binary_repr(s, len(slicing_edges_loop))))
        sliced_tensors = tensors.copy()
        edges_pool = slicing_edges_loop + [slicing_edges_companion_dict[edge] for edge in slicing_edges_loop if edge in slicing_edges_companion_dict.keys()]
        for i in range(len(edges_pool)):
            m, n = edges_pool[i]
            idxm_n, idxn_m = slicing_indices_dict[(m, n)]
            if i < len(slicing_edges_loop):
                j = i
            else:
                j =  slicing_edges_loop.index(
                    list(slicing_edges_companion_dict.keys())[list(slicing_edges_companion_dict.values()).index((m, n))]
                ) # i % len(slicing_edges_loop)
            sliced_tensors[m] = sliced_tensors[m].select(idxm_n, bitstring[j])
            sliced_tensors[n] = sliced_tensors[n].select(idxn_m, bitstring[j])
        
        collect_tensor = torch.zeros(simulation_data['shape_inter'], dtype=torch.complex64, device=device) # None #
        source = scheme[len(partition[0])-2][0][0]

        for s1 in range(tasks_front):
            if get_time:
                torch.cuda.synchronize(device)
                t0 = time.time()
            bitstring_front = list(map(int, np.binary_repr(s1, len(slicing_edges_front))))
            sliced_tensors_front = sliced_tensors.copy()
            edges_pool = slicing_edges_front + [slicing_edges_companion_dict[edge] for edge in slicing_edges_front if edge in slicing_edges_companion_dict.keys()]
            for i in range(len(edges_pool)):
                m, n = edges_pool[i]
                idxm_n, idxn_m = slicing_indices_dict[(m, n)]
                if i < len(slicing_edges_front):
                    j = i
                else:
                    j = slicing_edges_front.index(
                        list(slicing_edges_companion_dict.keys())[list(slicing_edges_companion_dict.values()).index((m, n))]
                    )# i % len(slicing_edges_front)
                sliced_tensors_front[m] = sliced_tensors_front[m].select(idxm_n, bitstring_front[j])
                sliced_tensors_front[n] = sliced_tensors_front[n].select(idxn_m, bitstring_front[j])
            # todevice(sliced_tensors_front, device)
            tensor_contraction_multibitstrings_test(sliced_tensors_front, scheme[:len(partition[0])-1])
            if collect_tensor is None:
                collect_tensor = sliced_tensors_front[source]
            else:
                collect_tensor += sliced_tensors_front[source]
            if get_time:
                torch.cuda.synchronize(device)
                print(f'{s1+1}/{2**len(slicing_edges_front)}', time.time() - t0)
        # collect_tensor = collect_tensor.cpu()
        sliced_tensors[source] = collect_tensor
        del collect_tensor
        del sliced_tensors_front
        source = scheme[-1][0][0]
        for s2 in range(tasks_back):
            if get_time:
                torch.cuda.synchronize(device)
                t1 = time.time()
            bitstring_back = list(map(int, np.binary_repr(s2, len(slicing_edges_back))))
            sliced_tensors_back = sliced_tensors.copy()
            # sliced_tensors_back[source] = collect_tensor.to(device)
            edges_pool = slicing_edges_back + [slicing_edges_companion_dict[edge] for edge in slicing_edges_back if edge in slicing_edges_companion_dict.keys()]
            for i in range(len(edges_pool)):
                m, n = edges_pool[i]
                if (m, n) not in slicing_indices_dict:
                    m = source
                idxm_n, idxn_m = slicing_indices_dict[(m, n)]
                if i < len(slicing_edges_back):
                    j = i
                else:
                    j = slicing_edges_back.index(
                        list(slicing_edges_companion_dict.keys())[list(slicing_edges_companion_dict.values()).index((m, n))]
                    )# i % len(slicing_edges_back)
                sliced_tensors_back[m] = sliced_tensors_back[m].select(idxm_n, bitstring_back[j])
                sliced_tensors_back[n] = sliced_tensors_back[n].select(idxn_m, bitstring_back[j])
            # todevice(sliced_tensors_back, device)
            tensor_contraction_multibitstrings_test(sliced_tensors_back, scheme[(len(partition[0])-1):])
            if collect_tensor_all is None:
                collect_tensor_all = sliced_tensors_back[source]
            else:
                collect_tensor_all += sliced_tensors_back[source]
            if get_time:
                torch.cuda.synchronize(device)
                print(f'{s2+1}/{2**len(slicing_edges_back)}', time.time() - t1)
        del sliced_tensors_back
    if get_time:
        torch.cuda.synchronize(device)
        t_end = time.time()
        print(f'overall time: {t_end-t_start:.2f}')

    return collect_tensor_all.permute(permute_idx)

def simulation(task_id, task_num, device, subtasks_num='all', get_time=False):
    simulation_data = torch.load(join(dirname(__file__), 'contraction_scheme.pt'))
    for i in range(len(simulation_data['tensors'])):
        simulation_data['tensors'][i] = simulation_data['tensors'][i].to(device)

    result = circuit_simulation_new(simulation_data, task_id, task_num, device, subtasks_num=subtasks_num, get_time=get_time)

    return result.cpu()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-cuda", type=int, default=0, help="cuda device to use, -1 for cpu")
    parser.add_argument("-task_start", type=int, default=0, help="start id of subtasks, each has 2**task_num subtasks")
    parser.add_argument("-task_end", type=int, default=1, help="end id of subtasks, each has 2**task_num subtasks")
    parser.add_argument("-task_num", type=int, default=0, help="# of subtasks for single run")
    parser.add_argument("-get_time", action='store_true', help="report the simulaiton time or not")
    parser.add_argument("-subtask_num", type=int, default=-1, help="# of front and back subtasks for one run, -1 for summing all")
    args = parser.parse_args()

    device = 'cpu' if args.cuda == -1 else f'cuda:{args.cuda}'
    subtasks_num = 'all' if args.subtask_num == -1 else args.subtask_num
    for task_id in range(args.task_start, args.task_end):
        print(f'subtask {task_id}:')
        samples = simulation(task_id, args.task_num, device, subtasks_num, args.get_time)