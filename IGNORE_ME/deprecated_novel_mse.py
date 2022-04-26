# def choose_unordered_subset(items: List[T], k: int) -> List[T]:
#     """ All unordered subsets (as lists) of `items`` of size `k`. The `choose` function is a helper. """
#     def choose(items: List[T], k: int, index: int) -> List[T]:
#         if len(items) - index < k:
#             return []
#         elif k == 1:
#             return [[items[i]] for i in range(len(items) - index, len(items), 1)]
#         else:
#             not_choosing: List[List[T]] = choose(items, k, index=index + 1)
#             choosing: List[List[T]] = choose(items, k - 1, index=index + 1)
#             for choice in choosing:
#                 choice.append(items[index])
#             return not_choosing + choosing
#     return choose(items, k, 0)


# def all_net_pairs():
#     combinations = choose_product([1, 2], 4)
#     assert len(combinations) == 16
#     mapping = {}
#     num = 1
#     for i in range(16):
#         for j in range(16):
#             mapping[num] = (
#                 "resnet"+"".join(map(lambda c: str(c), combinations[i]))+".pt",
#                 "resnet"+"".join(map(lambda c: str(c), combinations[j]))+".pt"
#             )
#             num += 1
#     return mapping, num


# class PairExp(object):
#     """ A part of an experiment that is for a pair of networks """

#     def __init__(
#             self: PairExp,
#             layers1: List[int],
#             layers2: List[int],
#     ) -> NoReturn:
#         # TODO
#         # 1. load the networks
#         # 2. load the random networks (controls)
#         # 3. create the stitch table
#         # 4. create the stitched network table
#         # 5. (below) use those tables and the traininer to
#         #    get the similarities
#         self.send_recv_sims = Table.mappedTable(None, None)
#         self.send_rand_recv_sims = Table.mappedTable(None, None)
#         self.send_recv_rand_sims = Table.mappedTable(None, None)
#         self.send_rand_recv_rand_sims = Table.mappedTable(None, None)

#     # TODO change the naming and actually use/test

#     @staticmethod
#     def mean2_multiple_stitches(
#         sender: Resnet,
#         reciever: Resnet,
#         stitch_tables: List[List[List[nn.Module]]],
#         labels: List[List[Tuple[LayerLabel, LayerLabel]]],
#         train_loader: DataLoader,
#     ) -> Tuple[Dict[Tuple[int, int], List[List[float]]], List[List[List[float]]]]:
#         """
#         Given a list of stitch tables, return the mean2 difference between
#         every pair of stitches pointwise for every pair of stitch tables (on the sender)
#         as well as the list of mean2 differences between each table in the stitch_tables
#         and the original network's expected representation (that of the reciever, output
#         from the layer before that recieving the stitch). The dictionary keys are the
#         indices of the labels list, while indices of the list of tables (comparing with
#         original network) correspond to the indices in the stitch tables.

#         Example Usage:
#         mean2_multiple_stitches(
#             sender,
#             reciever,
#             [vanilla_stitches, sim_stitches],
#             labels,
#             train_loader,
#         )
#         """

#         # 1. compare 2 stitches
#         #    - sender => sender, reciever => sender
#         #    - send_stitch is the stitch, recv_stitch is the other stitch
#         # 2. compare original with OG
#         #    - sender => sender, reciever => reciever
#         #    - send_stitch is the stitch
#         #    - recv_stitch is an identity function

#         # Choose all stitches and compare them to the original
#         identity = Identity()
#         identity_stitch_table: List[List[nn.Module]] = \
#             [[identity for _ in range(len(stitch_tables[0]))]
#              for _ in range(len(stitch_tables))]

#         mean2_original: List[List[List[float]]] = [
#             PairExp.mean2_model_model(
#                 sender,
#                 reciever,
#                 stitch_table,
#                 identity_stitch_table,
#                 labels,
#                 train_loader,
#             )
#             for stitch_table in stitch_tables
#         ]

#         # Choose all unordered pairs of stitches and compare them
#         stitch_pairs: List[List[List[List[nn.Module]]]] = \
#             choose_unordered_subset(stitch_tables, 2)
#         stitch_indices: List[List[int]] = \
#             choose_unordered_subset(list(range(len(stitch_tables))), 2)

#         mean2_stitches: Dict[Tuple[int, int], List[List[float]]] = {}
#         for stitch_pair, index_pair in zip(stitch_pairs, stitch_indices):
#             stitch_table1, stitch_table2 = stitch_pair
#             index1, index2 = index_pair
#             mean2_table = PairExp.mean2_model_model(
#                 sender,
#                 sender,
#                 stitch_table1,
#                 stitch_table2,
#                 labels,
#                 train_loader,
#             )
#             mean2_stitches[(index1, index2)] = mean2_table
#         return (mean2_stitches, mean2_original)

#     @ staticmethod
#     def mean2_model_model(
#         sender: Resnet,
#         reciever: Resnet,
#         send_stitches: List[List[nn.Module]],
#         recv_stitches: List[List[nn.Module]],
#         labels: List[List[Tuple[LayerLabel, LayerLabel]]],
#         train_loader: DataLoader
#     ) -> List[List[float]]:
#         assert len(labels) > 0
#         assert len(labels) == len(send_stitches)
#         assert len(labels) == len(recv_stitches)
#         assert len(labels[0]) > 0
#         assert len(labels[0]) == len(send_stitches[0])
#         assert len(labels[0]) == len(recv_stitches[0])

#         mean2_table: List[List[float]] = [
#             [0.0 for _ in range(len(labels[0]))] for _ in range(len(labels))]

#         for i in range(len(labels)):
#             for j in range(len(labels[i])):
#                 send_label, recv_label = labels[i][j]
#                 send_stitch, recv_stitch = send_stitches[i][j], recv_stitches[i][j]
#                 mean2_table[i][j] = PairExp.mean2_model_diff(
#                     send_stitch,
#                     recv_stitch,
#                     sender,
#                     reciever,
#                     send_label,
#                     recv_label,
#                     train_loader,
#                 )
#         return mean2_table

#     @staticmethod
#     def mean2_layer_layer(
#             send_stitch: nn.Module,
#             recv_stitch: nn.Module,
#             sender: Resnet,
#             reciever: Resnet,
#             send_label: LayerLabel,
#             recv_label: LayerLabel,
#             train_loader: DataLoader,
#     ) -> float:
#         num_images = len(train_loader)
#         total = 0.0
#         recv_label = recv_label - 1
#         for x, _ in train_loader:
#             with autocast():
#                 # TODO autocase and should we `pool_and_flatten``
#                 sent = sender.outfrom_forward(x, send_label)
#                 recv = reciever.outfrom_forward(x, recv_label)

#                 sent_stitch = send_stitch(sent)
#                 recv_stitch = recv_stitch(recv)

#                 diff = (sent_stitch - recv_stitch).pow(2).mean()
#                 total += diff.cpu().item()
#             pass
#         # Average over the number of images
#         total /= num_images
#         return total

# def choose_product(possibles: List[Any], length: int) -> List[List[Any]]:
#     """ All ordered subsequences of length `length` of where each element is in `possibles` """
#     if (length == 1):
#         return [[x] for x in possibles]
#     combinations = []
#     for possible in possibles:
#         remainders = choose_product(length - 1, possibles)
#         for remainder in remainders:
#             combinations.append(remainder + [possible])
#     return combinations