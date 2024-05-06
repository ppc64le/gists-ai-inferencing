# Deploying the Fraud Detection LSTM Model in OpenShift Container Platform

Contributors
----------------------
🎉🌟Thank you to all the contributors who contributed to this project: Revanth Atmakuri, Shadman Kaif, Theresa Xu.🎉🌟

Objective
----------------------
The objective of this project is to deploy the Fraud Detection LSTM model in an OpenShift Container Platform environment for efficient fraud detection.

Scope
----------------------
This project involves:

1. Building the image of Fraud Detection LSTM inferencing code using Podman.
2. Pushing the built image into the OpenShift Internal Registry.
3. Retrieving the image stored in the OpenShift Internal Registry.
4. Deploying the model using Kubernetes Deployment on OpenShift.
5. Configuring a service for the deployed model.
6. Setting up an ingress for the model endpoint.
7. Verifying the deployment's functionality and scalability.

This project will not cover the development or training of the Fraud Detection LSTM model itself, focusing solely on its deployment within an OpenShift environment.

How-To
----------------------
### Step 1: Build the image of Fraud Detection LSTM inferencing code using Podman
* Clone this repository and enter this directory, `cd lstm-workload-in-ocp`
* `podman build -t lstm-workload .`
* `podman images`
* Verify you have an image called localhost/lstm-workload.

### Step 2: Push the build image into the OpenShift Internal Registry
* `podman login -u kubeadmin -p $(oc whoami -t) --tls-verify=false default-route-openshift-image-registry.apps.domain.com`
* `podman tag localhost/lstm-workload:latest default-route-openshift-image-registry.apps.domain.com/lstmtesting/lstm-workload:latest`
* Make sure you see a new image with this name when run `podman images`
* `podman push default-route-openshift-image-registry.apps.domain.com/lstmtesting/lstm-workload --tls-verify=false`

### Step 3: Get the image stored in the OpenShift Internal Registry
* `oc get is`

### Step 4: Deployment 
Make sure you note the full path to the image in step 3, and edit the value called “image” in the deployment file to reflect the image path from step 3.
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lstm-endpoint
spec:
  replicas: 64
  selector:
    matchLabels:
      app: lstm-endpoint
  template:
    metadata:
      labels:
        app: lstm-endpoint
    spec:
      containers:
      - name: lstm-endpoint
        image: image-registry.openshift-image-registry.svc:5000/lstmtesting/lstm-workload@sha256:fe8b44105cd7c015fb329780f0f3692fb46ea02e566b239c8ebae66e434649a6
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 5000
        resources:
          limits:
            cpu: 1000m
            memory: 5Gi
          request:
            cpu: 1000m
            memory: 5Gi
```
* Deploy the pods by running `oc apply -f deployment_lstm_workload.yaml`

### Step 5: Deploy the service
```
apiVersion: v1
kind: Service
metadata:
  name: lstm-endpoint
  namespace: lstmtesting
spec:
  ports:
  - port: 5000
    protocol: TCP
    targetPort: 5000
  selector:
    app: lstm-endpoint
  type: NodePort
```
* Apply the changes by running `oc apply -f service_lstm_workload_nodeport.yaml`

### Step 6: Deploy the ingress
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lstm-endpoint-ingress
spec:
  rules:
  - http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: lstm-endpoint
            port:
              number: 5000
      - path: /testing
        pathType: Prefix
        backend:
          service:
            name: lstm-endpoint
            port:
              number: 5000
```
* To apply the changes, run `oc apply -f ingress_lstm_workload_nodeport.yaml`

### Step 7: Get the port number
* Run the command `oc get svc`
* An example output is: 
```
#Eg: [root@bastion lstm-workload-in-ocp]# oc get svc
#NAME            TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
#lstm-endpoint   NodePort   YOUR IP          <none>        5000:30915/TCP   2d23h
```
From this, the port number would be the second port - 30915.

### Step 8: Verify the deployment
* `oc get pod | grep lstm-endpoint | wc -l`
* Make sure the above output give the number reflecting the number of replicas by running:
  
```
[root@bastion lstm-workload-in-ocp]# curl -X POST -H "Content-Type: application/json" -d '{"user_id":0, "card":0, "year":2002, "month":9, "day": 13, "time": "06:37", "amount": "$44.41", "use_chip": "Swipe Transaction", "merchant_name": "-34551508091458520", "merchant_city": " ONLINE", "merchant_state": "CA", "zip_code": 91750, "mcc": 5912}' http://ai-w1.ai.toropsp.com:30915/predict
{
  "predictions": [
    0.45657798647880554
  ]
}
```

### Steps to Scale the pods or reallocate the resources
In order to scale the pods, you can open the deployment file and change the values respective to your needs:
```
[root@bastion lstm-workload-in-ocp]# oc get deployment
NAME                     READY   UP-TO-DATE   AVAILABLE   AGE
lstm-endpoint   64/64   64           64          46h
[root@bastion lstm-workload-in-ocp]# oc edit deployment lstm-endpoint
```
Here is the yaml file that's opened which you can change to reflect your resource requirements:
```
# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: "4"
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"name":"lstm-endpoint","namespace":"lstmtesting"},"spec":{"replicas":1,"selector":{"matchLabels":{"app":"lstm-endpoint"}},"template":{"metadata":{"labels":{"app":"lstm-endpoint"}},"spec":{"containers":[{"image":"image-registry.openshift-image-registry.svc:5000/lstmtesting/lstm-workload@sha256:fe8b44105cd7c015fb329780f0f3692fb46ea02e566b239c8ebae66e434649a6","imagePullPolicy":"IfNotPresent","limits":{"cpu":"100m","memory":"4Gi"},"name":"lstm-endpoint","ports":[{"containerPort":5000}],"request":{"cpu":"100m","memory":"4Gi"},"resources":null}]}}}}
  creationTimestamp: "2024-04-28T15:31:59Z"
  generation: 14
  name: lstm-endpoint
  namespace: lstmtesting
  resourceVersion: "2244148"
  uid: 38de7f1b-5912-4820-aac5-2e157ff19a50
spec:
  progressDeadlineSeconds: 600
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: lstm-endpoint
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: lstm-endpoint
    spec:
      containers:
      - image: image-registry.openshift-image-registry.svc:5000/lstmtesting/lstm-workload@sha256:fe8b44105cd7c015fb329780f0f3692fb46ea02e566b239c8ebae66e434649a6
        imagePullPolicy: IfNotPresent
        name: lstm-endpoint
        ports:
        - containerPort: 5000
          protocol: TCP
        resources: {}
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
```
### Testing with JMeter
The Apache JMeter™ application is open source software, a 100% pure Java application designed to load test functional behavior and measure performance. It was originally designed for testing Web Applications but has since expanded to other test functions. You can download JMeter from https://jmeter.apache.org/download_jmeter.cgi. Ensure you have Java 8+ installed on your system.
We have included a JMeter file in this repository: OCP-fraud-detection-power-final.jmx. Run JMeter on your system and change the inferencing endpoint to reflect your created endpoint.
