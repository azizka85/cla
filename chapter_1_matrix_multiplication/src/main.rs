use matrix_multiplication::*;

fn main() {
    let x = vec![1., 2., 3.];
    let y = vec![3., 7., 9.];

    let a = 2.;

    println!("Dot product:");
    println!("{:?} * {:?} = {}", x, y, dot_product(&x, &y));

    let mut z = y.clone();

    saxpy(a, &x, &mut z);

    println!("\nSaxpy:");
    println!("{a}*{:?} + {:?} = {:?}", x, y, z);

    let a = vec![
        vec![1., 2.],
        vec![3., 4.],
        vec![5., 6.]
    ];

    println!("\nA = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let b = transpose(&a);

    println!("\nTranspose of A = [");

    for i in 0..b.len() {
        println!("  {:?}", b[i]);
    }

    println!("]");

    let x = vec![1., 2.];
    let y = vec![3., 7., 9.];

    let mut r = y.clone();
    let mut c = y.clone();

    row_oriented_gaxpy(&a, &x, &mut r);
    column_oriented_gaxpy(&a, &x, &mut c);    

    println!("\nRow oriented Gaxpy:");
    println!("A*{:?} + {:?} = {:?}", x, y, r);

    println!("\nColumn oriented Gaxpy:");
    println!("A*{:?} + {:?} = {:?}", x, y, c);

    let a = vec![
        vec![1., 2., 3.],
        vec![4., 5., 6.]
    ];

    println!("\nA = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let mut b = a.clone();

    row_oriented_outer_product_update(&mut b, &x, &y);

    println!("\nRow oriented outer product update:");
    println!("A + {:?} * {:?} = [", x, y);

    for i in 0..b.len() {
        println!("  {:?}", b[i]);
    }

    println!("]");

    let mut c = a.clone();

    column_oriented_outer_product_update(&mut c, &x, &y);

    println!("\nColumn oriented outer product update:");
    println!("A + {:?} * {:?} = [", x, y);

    for i in 0..b.len() {
        println!("  {:?}", c[i]);
    }

    println!("]");

    let a = vec![
        vec![1., 2.],
        vec![3., 4.]
    ];

    println!("\nA = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let b = vec![
        vec![5., 6.],
        vec![7., 8.]
    ];

    println!("\nB = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let bt = transpose(&b);

    let mut r1 = vec![vec![0.; bt.len()]; a.len()];

    for i in 0..a.len() {
        for j in 0..bt.len() {
            r1[i][j] = dot_product(&a[i], &bt[j])
        }
    }

    println!("\nDot product version of matrix multiplication:");
    println!("A * B = [");

    for i in 0..r1.len() {
        println!("  {:?}", r1[i]);
    }

    println!("]");

    let at = transpose(&a);

    let mut r2 = vec![vec![0.; a.len()]; bt.len()];

    for i in 0..bt.len() {
        for j in 0..bt[i].len() {
            saxpy(bt[i][j], &at[j], &mut r2[i])
        }
    }

    let r2 = transpose(&r2);

    println!("\nSaxpy version of matrix multiplication:");
    println!("A * B = [");

    for i in 0..r2.len() {
        println!("  {:?}", r2[i]);
    }

    println!("]");

    let mut r3 = vec![vec![0.; bt.len()]; a.len()];

    for i in 0..b.len() {
        row_oriented_outer_product_update(&mut r3, &at[i], &b[i]);
    }

    println!("\nOuter product version of matrix multiplication:");
    println!("A * B = [");

    for i in 0..r3.len() {
        println!("  {:?}", r3[i]);
    }

    println!("]");

    let a = vec![
        vec![ 1.,  2.,  3.,  0.,  0.,  0.],
        vec![ 4.,  5.,  6.,  7.,  0.,  0.],
        vec![ 0.,  8.,  9., 10., 11.,  0.],
        vec![ 0.,  0., 12., 13., 14., 15.],
        vec![ 0.,  0.,  0., 16., 17., 18.],
        vec![ 0.,  0.,  0.,  0., 19., 20.]
    ];

    println!("\nA = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let x = vec![1.; 6];
    let y = vec![0.; 6];

    let mut r = y.clone();
    
    row_oriented_gaxpy(&a, &x, &mut r);    

    println!("\nRow oriented Gaxpy:");
    println!("A*{:?} + {:?} = {:?}", x, y, r);

    let b = vec![
        vec![ 0.,  0.,  3.,  7., 11., 15.],
        vec![ 0.,  2.,  6., 10., 14., 18.],
        vec![ 1.,  5.,  9., 13., 17., 20.],
        vec![ 4.,  8., 12., 16., 19.,  0.]
    ];

    let mut r = y.clone();

    band::band_gaxpy(&b, 1, 2, &x, &mut r);

    println!("\nBand Storage Gaxpy:");
    println!("A*{:?} + {:?} = {:?}", x, y, r);

    let b = vec![2, 3, 1];
    let mut r = y.clone();

    block::row_blocked_gaxpy(&a, &b, &x, &mut r);

    println!("\nRow-blocked Gaxpy with blocks {:?}:", b);
    println!("A*{:?} + {:?} = {:?}", x, y, r);

    let b = vec![1, 3, 2];
    let mut r = y.clone();

    block::column_blocked_gaxpy(&a, &b, &x, &mut r);

    println!("\nColumn-blocked Gaxpy with blocks {:?}:", b);
    println!("A*{:?} + {:?} = {:?}", x, y, r);

    let a = vec![
        vec![ 1.,  2.,  3.,  0.,  0.,  0.],
        vec![ 0.,  5.,  6.,  7.,  0.,  0.],
        vec![ 0.,  0.,  9., 10., 11.,  0.],
        vec![ 0.,  0.,  0., 13., 14., 15.],
        vec![ 0.,  0.,  0.,  0., 17., 18.],
        vec![ 0.,  0.,  0.,  0.,  0., 20.]
    ];

    println!("\nA = [");

    for i in 0..a.len() {
        println!("  {:?}", a[i]);
    }

    println!("]");

    let b = vec![
        vec![ 1.,  2.,  3.,  0.,  0.,  0.],
        vec![ 0.,  5.,  6.,  7.,  0.,  0.],
        vec![ 0.,  0.,  9., 10., 11.,  0.],
        vec![ 0.,  0.,  0., 13., 14., 15.],
        vec![ 0.,  0.,  0.,  0., 17., 18.],
        vec![ 0.,  0.,  0.,  0.,  0., 20.]
    ];

    println!("\nB = [");

    for i in 0..b.len() {
        println!("  {:?}", b[i]);
    }

    println!("]");

    let mut c = vec![vec![0.; 6]; 6];
    let at = transpose(&a);

    for i in 0..b.len() {
        row_oriented_outer_product_update(&mut c, &at[i], &b[i]);
    }

    println!("\nOuter product version of matrix multiplication:");
    println!("C = A * B = [");

    for i in 0..c.len() {
        println!("  {:?}", c[i]);
    }

    println!("]");

    let mut c = vec![vec![0.; 6]; 6];

    block::block_update(&mut c, &a, &b, 2, 3);

    println!("\nBlock matrix multiplication:");
    println!("C = A*B = [");

    for i in 0..c.len() {
        println!("  {:?}", c[i]);
    }

    println!("]");
}
