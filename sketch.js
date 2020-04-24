
const model = tf.sequential();
// tfvis.visor().surface({name: 'My Surface', tab: 'My Tab'});
const confighidden = {
    units: 4,
    inputShape: [2],
    activation:'sigmoid'
}
const hidden = tf.layers.dense(confighidden);

const configoutput = {
    units: 1 ,
    activation: 'sigmoid'
}
const output = tf.layers.dense(configoutput);

model.add(hidden);
model.add(output);

const surface = { name: 'Layer Summary', tab: 'Model Inspection'};
tfvis.show.layer(surface, model.getLayer(undefined, 1));

const sgdop = tf.train.sgd(0.1)
const configopt = {
    optimizer:sgdop,
    loss: 'meanSquaredError'
}

const xs = tf.tensor2d([
    [0,0],
    [0.5,0.5],
    [1,1]
]);

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
]);
model.compile(configopt);

train().then(() => {
    let outputs = model.predict(xs);
    outputs.print();
    console.log('training complete');
});

async function train(){
    for(let i=0;i<1000;i++){
        const config = {
            shuffle: true,
            epochs:10
        }
        // tfvis.visor().surface({name: 'My Surface', tab: 'My Tab'});
        const response = await model.fit(xs,ys,config);
        console.log(response.history.loss[0]);
    }
}



