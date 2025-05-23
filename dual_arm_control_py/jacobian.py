import numpy as np

def compute_jacobian(theta, dh_l):
    theta = np.array(theta)
    l1,l2,l3,l4,l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4]
    
    t1 = theta[0] + np.pi / 2
    t2 = theta[1]
    t3 = theta[2]
    t4 = theta[3]
    t5 = theta[4]
    t6 = theta[5]

    sin_t1 = np.sin(t1)
    cos_t1 = np.cos(t1)
    sin_t2 = np.sin(t2)
    cos_t2 = np.cos(t2)
    sin_t3 = np.sin(t3)
    cos_t3 = np.cos(t3)
    sin_t4 = np.sin(t4)
    cos_t4 = np.cos(t4)
    sin_t5 = np.sin(t5)
    cos_t5 = np.cos(t5)
    sin_t6 = np.sin(t6)
    cos_t6 = np.cos(t6)

    expr1 = sin_t1

    expr2 = -cos_t1 * sin_t2

    expr3 = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3

    expr4 = (
        sin_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
        - cos_t4 * cos_t1 * sin_t2
    )

    expr5 = (
        cos_t5 * (cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3)
        - sin_t5 * (
            cos_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
            + cos_t1 * sin_t2 * sin_t4
        )
    )

    expr6 = (
        cos_t6 * (
            sin_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
            - cos_t4 * cos_t1 * sin_t2
        )
        + sin_t6 * (
            cos_t5 * (
                cos_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
                + cos_t1 * sin_t2 * sin_t4
            )
            + sin_t5 * (cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3)
        )
    )

    
    ####row 2
    expr21 = -1
    expr22 = 0
    expr23 = -cos_t2
    expr24 = -sin_t2 * sin_t3
    expr25 = cos_t3 * sin_t2 * sin_t4 - cos_t2 * cos_t4
    expr26 = -sin_t5 * (cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2) - cos_t5 * sin_t2 * sin_t3
    expr27 = (
        sin_t6 * (cos_t5 * (cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2) - sin_t2 * sin_t3 * sin_t5)
        - cos_t6 * (cos_t2 * cos_t4 - cos_t3 * sin_t2 * sin_t4)
    )


    #### row3
    expr31 = 0
    expr32 = -cos_t1
    expr33 = -sin_t2 * sin_t1
    expr34 = cos_t2 * sin_t3 * sin_t1 - cos_t3 * cos_t1
    expr35 = -sin_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1) - cos_t4 * sin_t2 * sin_t1

    expr36 = (
        sin_t5 * (
            cos_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
            - sin_t2 * sin_t4 * sin_t1
        )
        - cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
    )

    expr37 = (
        -sin_t6 * (
            cos_t5 * (
                cos_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
                - sin_t2 * sin_t4 * sin_t1
            )
            + sin_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
        )
        - cos_t6 * (
            sin_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
            + cos_t4 * sin_t2 * sin_t1
        )
    )



    #### row4
    A = cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1
    B = cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1
    C = sin_t4 * A + cos_t4 * sin_t2 * sin_t1
    D = cos_t4 * A - sin_t2 * sin_t4 * sin_t1
    E = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3
    F = sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1
    G = sin_t4 * F - cos_t4 * cos_t1 * sin_t2
    H = cos_t4 * F + cos_t1 * sin_t2 * sin_t4

    expr41 = l4 * C + l5 * (sin_t6 * (cos_t5 * D + sin_t5 * B) + cos_t6 * C) + l3 * sin_t2 * sin_t1

    expr42 = l5 * (
        sin_t6 * (
            cos_t5 * (cos_t2 * cos_t1 * sin_t4 + cos_t3 * cos_t4 * cos_t1 * sin_t2)
            - cos_t1 * sin_t2 * sin_t3 * sin_t5
        )
        - cos_t6 * (cos_t2 * cos_t4 * cos_t1 - cos_t3 * cos_t1 * sin_t2 * sin_t4)
    ) - l4 * (cos_t2 * cos_t4 * cos_t1 - cos_t3 * cos_t1 * sin_t2 * sin_t4) - l3 * cos_t2 * cos_t1

    expr43 = l4 * sin_t4 * E - l5 * (
        sin_t6 * (
            sin_t5 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
            - cos_t4 * cos_t5 * E
        )
        - cos_t6 * sin_t4 * E
    )

    expr44 = l5 * (
        cos_t6 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)
        - cos_t5 * sin_t6 * (sin_t4 * F - cos_t4 * cos_t1 * sin_t2)
    ) + l4 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)

    expr45 = -l5 * sin_t6 * (
        sin_t5 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)
        - cos_t5 * E
    )

    expr46 = -l5 * (
        sin_t6 * G
        - cos_t6 * (cos_t5 * H + sin_t5 * E)
    )

    expr47 = 0


    ##### row5
    term1 = cos_t4 * sin_t2 + cos_t2 * cos_t3 * sin_t4
    term2 = sin_t2 * sin_t4 - cos_t2 * cos_t3 * cos_t4
    term3 = cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2
    term4 = cos_t2 * cos_t4 - cos_t3 * sin_t2 * sin_t4

    # Compute output vector elements
    expr51 = 0.0

    expr52 = -l5 * (sin_t6 * (cos_t3 * sin_t2 * sin_t5 + cos_t4 * cos_t5 * sin_t2 * sin_t3) + cos_t6 * sin_t2 * sin_t3 * sin_t4) - l4 * sin_t2 * sin_t3 * sin_t4

    expr53 = l4 * term3 + l5 * (cos_t6 * term3 + cos_t5 * sin_t6 * term4)

    expr54 = -l5 * sin_t6 * (sin_t5 * term3 + cos_t5 * sin_t2 * sin_t3)

    expr55 = l5 * (cos_t6 * (cos_t5 * term3 - sin_t2 * sin_t3 * sin_t5) + sin_t6 * term4)

    expr56 = 0

    expr57 = 0

    #### row6
    A = sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1
    B = cos_t4 * cos_t1 * sin_t2
    C = sin_t4 * A - B
    D = cos_t4 * A + cos_t1 * sin_t2 * sin_t4
    E = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3

    F = cos_t2 * sin_t4 * sin_t1 + cos_t3 * cos_t4 * sin_t2 * sin_t1
    G = cos_t2 * cos_t4 * sin_t1 - cos_t3 * sin_t2 * sin_t4 * sin_t1

    H = cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1
    I = sin_t2 * sin_t4 * sin_t1

    # Compute outputs
    expr61 = (
        l4 * C
        + l5 * (cos_t6 * C + sin_t6 * (cos_t5 * D + sin_t5 * E))
        - l3 * cos_t1 * sin_t2
    )

    expr62 = (
        l5
        * (
            sin_t6
            * (
                cos_t5 * F
                - sin_t2 * sin_t3 * sin_t5 * sin_t1
            )
            - cos_t6 * G
        )
        - l4 * G
        - l3 * cos_t2 * sin_t1
    )

    expr63 = (
        l5
        * (
            sin_t6
            * (
                sin_t5 * H
                - cos_t4 * cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
            )
            - cos_t6 * sin_t4 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
        )
        - l4 * sin_t4 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
    )

    expr64 = -(
        l4
        * (
            cos_t4 * H
            - sin_t2 * sin_t4 * sin_t1
        )
        + l5
        * (
            cos_t6
            * (
                cos_t4 * H
                - sin_t2 * sin_t4 * sin_t1
            )
            - cos_t5 * sin_t6
            * (
                sin_t4 * H
                + cos_t4 * sin_t2 * sin_t1
            )
        )
    )

    expr65 = (
        l5
        * sin_t6
        * (
            sin_t5
            * (
                cos_t4 * H
                - sin_t2 * sin_t4 * sin_t1
            )
            - cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
        )
    )

    expr66 = (
        l5
        * (
            sin_t6
            * (
                sin_t4 * H
                + cos_t4 * sin_t2 * sin_t1
            )
            - cos_t6
            * (
                cos_t5
                * (
                    cos_t4 * H
                    - sin_t2 * sin_t4 * sin_t1
                )
                + sin_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
            )
        )
    )

    expr67 = 0


    return np.array([[0,expr1, expr2, expr3, expr4, expr5, expr6],
                     [expr21, expr22, expr23, expr24, expr25, expr26,expr27],
                     [expr31, expr32, expr33, expr34, expr35, expr36,expr37],
                     [expr41, expr42, expr43, expr44, expr45, expr46,expr47],
                     [expr51, expr52, expr53, expr54, expr55, expr56,expr57],
                     [expr61, expr62, expr63, expr64, expr65, expr66,expr67]])
print(compute_jacobian([np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7], [0.10555,0.176,0.3,0.32,0.2251]))