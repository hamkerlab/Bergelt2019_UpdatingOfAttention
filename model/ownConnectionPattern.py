"""
@author: juschu

defining own connection pattern
"""


##############################
#### imports and settings ####
##############################
import math

from ANNarchy import CSR

from auxFunctions_model import ProgressOutput

MIN_CONNECTION_VALUE = 0.001
fullOutput = False


############################
#### connection pattern ####
############################
def all2all_exp2d(pre, post, factor, radius, mgd):
    '''
    connecting two 2-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            factor    -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("all2all_exp2d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1])
    postDimLength = (post.geometry[0], post.geometry[1])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (preDimLength[0]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2 in range(postDimLength[1]):
            pre_ranks = []
            values = []

            for pre1 in range(preDimLength[0]):
                for pre2 in range(preDimLength[1]):
                    # distance between 2 neurons
                    dist_h = (post1-pre1)*(post1-pre1)
                    dist_v = (post2-pre2)*(post2-pre2)

                    if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                        # weight of connection
                        val = factor * m_exp(-((dist_h+dist_v)/sigma/sigma))

                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            numOfConnectionsCreated += 1
                            pre_rank = pre.rank_from_coordinates((pre1, pre2))
                            pre_ranks.append(pre_rank)
                            values.append(val)

            post_rank = post.rank_from_coordinates((post1, post2))
            synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)

    return synapses

def all2all_exp4d(pre, post, factor, radius, mgd):
    '''
    connecting two 4-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            factor    -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("all2all_exp4d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (preDimLength[0]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/factor))

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2 in range(postDimLength[1]):
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):
                    pre_ranks = []
                    values = []

                    # for speedup
                    rks_app = pre_ranks.append
                    vals_app = values.append
                    min1 = max(0, int(math.ceil(post1-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(post1+max_dist)))
                    min2 = max(0, int(math.ceil(post2-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(post2+max_dist)))
                    min3 = max(0, int(math.ceil(post3-max_dist)))
                    max3 = min(preDimLength[2]-1, int(math.floor(post3+max_dist)))
                    min4 = max(0, int(math.ceil(post4-max_dist)))
                    max4 = min(preDimLength[3]-1, int(math.floor(post4+max_dist)))

                    # faster iteration in for-loops
                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                            for pre3 in range(min3, max3+1): #range(preDimLength[2]):
                                for pre4 in range(min4, max4+1): #range(preDimLength[3]):
                                    # distance between 2 neurons
                                    dist_h1 = (post1-pre1)*(post1-pre1)
                                    dist_h2 = (post2-pre2)*(post2-pre2)
                                    dist_v1 = (post3-pre3)*(post3-pre3)
                                    dist_v2 = (post4-pre4)*(post4-pre4)

                                    if (mgd == 0) or (dist_h1 < mgd*mgd and dist_h2 < mgd*mgd and dist_v1 < mgd*mgd and dist_v2 < mgd*mgd):
                                        # weight of connection
                                        val = (factor * m_exp(-((dist_h1+dist_h2)/sigma/sigma +
                                                                (dist_v1+dist_v2)/sigma/sigma)))

                                        if val > MIN_CONNECTION_VALUE:
                                            # connect
                                            numOfConnectionsCreated += 1
                                            pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3,
                                                                                  pre4))
                                            rks_app(pre_rank)
                                            vals_app(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)

    return synapses

def gaussian2dTo4d_h(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 2d to 4d
    independent of last two dimension of 4d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian2dTo4d_h", pre, post)

    #                     w                h
    preDimLength = (pre.geometry[0], pre.geometry[1])
    #                       w1                w2                h1                h2
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (preDimLength[0]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    #w_post
    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2 in range(postDimLength[1]):

            # faster iteration in for-loops
            min1 = max(0, int(math.ceil(post1-max_dist)))
            max1 = min(preDimLength[0]-1, int(math.floor(post1+max_dist)))
            min2 = max(0, int(math.ceil(post2-max_dist)))
            max2 = min(preDimLength[1]-1, int(math.floor(post2+max_dist)))

            #h_post
            for post3  in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):
                    pre_ranks = []
                    values = []

                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                            # distance between 2 neurons
                            dist_h = (post1-pre1)*(post1-pre1)
                            dist_v = (post2-pre2)*(post2-pre2)

                            if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                                # weight of connection
                                val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                                if val > MIN_CONNECTION_VALUE:
                                    # connect
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)

    return synapses

def gaussian2dTo4d_v(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 2d to 4d
    independent of first two dimension of 4d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian2dTo4d_v", pre, post)

    #                      w                h
    preDimLength = (pre.geometry[0], pre.geometry[1])
    #                       w1                w2                h1                h2
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (preDimLength[0]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    #w_post
    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2  in range(postDimLength[1]):
            #h_post
            for post3  in range(postDimLength[2]):
                for post4  in range(postDimLength[3]):
                    pre_ranks = []
                    values = []

                    # for speedup
                    min1 = max(0, int(math.ceil(post3-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(post3+max_dist)))
                    min2 = max(0, int(math.ceil(post4-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(post4+max_dist)))

                    # faster iteration in for-loops
                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                            # distance between 2 neurons
                            dist_h = (post3-pre1)*(post3-pre1)
                            dist_v = (post4-pre2)*(post4-pre2)

                            if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                                # weight of connection
                                val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                                if val > MIN_CONNECTION_VALUE:
                                    # connect
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)

    return synapses

def gaussian4dTo2d_h(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 4d to 2d
    independent of last two dimensions of 4d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian4dTo2d_h", pre, post)

    #                     w1               w2               h1               h2
    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    #                       w                 h
    postDimLength = (post.geometry[0], post.geometry[1])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (preDimLength[0]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    # w_post
    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2  in range(postDimLength[1]):
            pre_ranks = []
            values = []

            # for speedup
            min1 = max(0, int(math.ceil(post1-max_dist)))
            max1 = min(preDimLength[0]-1, int(math.floor(post1+max_dist)))
            min2 = max(0, int(math.ceil(post2-max_dist)))
            max2 = min(preDimLength[1]-1, int(math.floor(post2+max_dist)))

            # w_pre
            # faster iteration in for-loops
            for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                    # distance between 2 neurons
                    dist_h = (post1-pre1)*(post1-pre1)
                    dist_v = (post2-pre2)*(post2-pre2)

                    if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                        # weight of connection
                        val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                        if val > MIN_CONNECTION_VALUE:
                            # h_pre
                            for pre3  in range(preDimLength[2]):
                                for pre4  in range(preDimLength[3]):
                                    # connect
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

            post_rank = post.rank_from_coordinates((post1, post2))
            synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)
    return synapses

def gaussian2dTo4d_diag(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 2d to 4d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian2dTo4d_diag", pre, post)

    #                     w                h
    preDimLength = (pre.geometry[0], pre.geometry[1])
    #                       w1                h1                w2                h2
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (postDimLength[2]-1)
    offset_h = (preDimLength[0]-1)/2.0
    offset_v = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numConnectionCreated = 0

    # w_post
    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numConnectionCreated, fullOutput)

        for post2 in range(postDimLength[1]):
            # h_post
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):
                    pre_ranks = []
                    values = []

                    # for speedup
                    min1 = max(0, int(math.ceil(post1+post3-offset_h-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(post1+post3-offset_h+max_dist)))
                    min2 = max(0, int(math.ceil(post2+post4-offset_v-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(post2+post4-offset_v+max_dist)))

                    # w_pre
                    # faster iteration in for-loops
                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                            # distance between 2 neurons
                            dist_h = (pre1-post1-post3+offset_h)*(pre1-post1-post3+offset_h)
                            dist_v = (pre2-post2-post4+offset_v)*(pre2-post2-post4+offset_v)

                            if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                                # weight of connection
                                val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                                if val > MIN_CONNECTION_VALUE:
                                    # connect
                                    numConnectionCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numConnectionCreated, fullOutput)

    return synapses

def gaussian4dTo2d_diag(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 4d to 2d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian4dTo2d_diag", pre, post)

    #                     w1               w2               h1               h2
    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    #                       w                 h
    postDimLength = (post.geometry[0], post.geometry[1])

    # Normalization along width of sigma values on afferent map
    # not very consistent (properly better to normalize along diagonal)
    sigma = radius * (preDimLength[2]-1)
    offset_h = (preDimLength[0]-1)/2.0
    offset_v = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    # w_post
    for post1 in range(postDimLength[0]):
        dout.print_conn(post1, postDimLength[0], numOfConnectionsCreated, fullOutput)

        for post2  in range(postDimLength[1]):
            pre_ranks = []
            values = []

            # w_pre
            for pre1 in range(preDimLength[0]):
                for pre2 in range(preDimLength[1]):

                    # for speedup
                    min1 = max(0, int(math.ceil(post1+offset_h-pre1-max_dist)))
                    max1 = min(preDimLength[2]-1, int(math.floor(post1+offset_h-pre1+max_dist)))
                    min2 = max(0, int(math.ceil(post2+offset_v-pre2-max_dist)))
                    max2 = min(preDimLength[3]-1, int(math.floor(post2+offset_v-pre2+max_dist)))

                    # h_pre
                    # faster iteration in for-loops
                    for pre3 in range(min1, max1+1): #range(preDimLength[2]):
                        for pre4 in range(min2, max2+1): #range(preDimLength[3]):
                            # distance between 2 neurons
                            dist_h = (post1-pre1-pre3+offset_h)*(post1-pre1-pre3+offset_h)
                            dist_v = (post2-pre2-pre4+offset_v)*(post2-pre2-pre4+offset_v)

                            if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                                # weight of connection
                                val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                                if val > MIN_CONNECTION_VALUE:
                                    # connect
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

            post_rank = post.rank_from_coordinates((post1, post2))
            synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numOfConnectionsCreated, fullOutput)

    return synapses

def gaussian4d_diagTo4d_v(pre, post, mv, radius, mgd):
    '''
    connect two maps with a gaussian receptive field 4d to 4d diagonally
    independent of first two dimension of post map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
                         0, if no limitation

    return: synapses  -- CSR-object with connections
    '''

    dout = ProgressOutput("gaussian4d_diagTo4d_v", pre, post)

    #                     w1               w2               h1               h2
    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    #                       w1                w2                h1                h2
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Normalization along width of sigma values on afferent map
    sigma = radius * (postDimLength[2]-1)
    offset_h = (preDimLength[0]-1)/2.0
    offset_v = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numConnectionCreated = 0

    #w_post
    for post1 in range(postDimLength[0]):
        for post2 in range(postDimLength[1]):
            dout.print_conn(post1*postDimLength[1]+post2, postDimLength[0]*postDimLength[1],
                            numConnectionCreated, fullOutput)

            #h_post
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):
                    pre_ranks = []
                    values = []

                    # w_pre
                    for pre1 in range(preDimLength[0]):
                        for pre2 in range(preDimLength[1]):
                            # for speedup
                            min1 = max(0, int(math.ceil(post3+offset_h-pre1-max_dist)))
                            max1 = min(preDimLength[2]-1,
                                       int(math.floor(post3+offset_h-pre1+max_dist)))
                            min2 = max(0, int(math.ceil(post4+offset_v-pre2-max_dist)))
                            max2 = min(preDimLength[3]-1,
                                       int(math.floor(post4+offset_v-pre2+max_dist)))

                            # h_pre
                            # faster iteration in for-loops
                            for pre3 in range(min1, max1+1): #range(preDimLength[2]):
                                for pre4 in range(min2, max2+1): #range(preDimLength[3]):
                                    # distance between 2 neurons
                                    dist_h = (pre1+pre3-post3-offset_h)*(pre1+pre3-post3-offset_h)
                                    dist_v = (pre2+pre4-post4-offset_v)*(pre2+pre4-post4-offset_v)

                                    if (mgd == 0) or (dist_h < mgd*mgd and dist_v < mgd*mgd):
                                        # weight of connection
                                        val = mv * m_exp(-((dist_h+dist_v)/sigma/sigma))

                                        if val > MIN_CONNECTION_VALUE:
                                            # connect
                                            numConnectionCreated += 1
                                            pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3,
                                                                                  pre4))
                                            pre_ranks.append(pre_rank)
                                            values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.print_conn(1, 1, numConnectionCreated, fullOutput)

    return synapses
