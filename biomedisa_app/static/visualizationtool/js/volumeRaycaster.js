(function(namespace) {

	namespace.VRC = {};
})(window);

(function(namespace) {
    var EventDispatcher = function(O) {

        var me = {};
        me.class = this;

        me._functions = [];
        me._context = window;

        me.add = function(func, is_start) {
            me._functions.push({"is_start": is_start != undefined ? is_start : true, "func": func});
            return me._functions.length-1;

        };

        me.remove = function(index) {
            delete( me._functions[index] );
        };

        me.get = function(index) {
            return me._functions[index];
        };

        me.stop = function(index) {
            me._functions[index]["is_start"] = false;
        };

        me.start = function(index) {
            me._functions[index]["is_start"] = true;
        };

        me.isStart = function(index) {
            return me._functions[index]["is_start"];
        };

        me.call = function(value, context) {
            var context = context ? context : me._context;

            for(i in me._functions) {
                var task = me._functions[i];
                if(task["is_start"]) {
                    task["func"].call(context, value);

                }
            };

        };

        me.isEmpty = function() {
            return me._functions.length == 0;

        };

        me.setConfig = function(O) {
            for(prop in O) {
                switch(prop) {
                    case "context": {
                        this._context = O[prop];
                    };break;

                };    
            };

        };

        /**
        * Constructor
        *
        * @method EventDispatcher.Constructor
        * @this {RC.EventDispatcher}
        * @EventDispatcher {Object} O
        * @EventDispatcher {Object} O.self         Context for calling
        */
        me.Constructor = function(O) {
            this.setConfig(O);
            
        };

        me.Constructor(O);

        return me;

    };
    
    namespace.EventDispatcher = EventDispatcher;

})(window.VRC);

(function(namespace) {
    var AdaptationManager = function() {

        var me = {};

        me._core = {};

        me._step = 5;
        me._steps = me._step * 2;

        me._onPostDrawFuncIndex = -1;
        me._onCameraChangeStartFuncIndex = -1;
        me._onCameraChangeEndFuncIndex = -1;

        me.init = function(core) {
            me._core = core;

            me._onPostDrawFuncIndex = me._core.onPostDraw.add(function(fps) {
                me.do(fps);
            });

            me._onCameraChangeStartFuncIndex = me._core.onCameraChangeStart.add(function() {
                // me.pause(true);

            });

            me._onCameraChangeEndFuncIndex = me._core.onCameraChangeEnd.add(function() {
                // me.pause(false);
            });


        };

        me.run = function(flag) {
            if(flag) {
                me._core.onPostDraw.start(me._onPostDrawFuncIndex);
                me._core.onCameraChangeStart.start(me._onCameraChangeEndFuncIndex);
                me._core.onCameraChangeEnd.start(me._onCameraChangeStartFuncIndex);

            } else {
                me._core.onPostDraw.stop(me._onPostDrawFuncIndex);
                me._core.onCameraChangeStart.stop(me._onCameraChangeEndFuncIndex);
                me._core.onCameraChangeEnd.stop(me._onCameraChangeStartFuncIndex);
               
            }

        };

        me.pause = function(flag) {
            if(flag) {
                me._core.onCameraChangeStart.stop(me._onCameraChangeEndFuncIndex);
                me._core.onPostDraw.stop(me._onPostDrawFuncIndex);
             

            } else {
                me._core.onCameraChangeStart.start(me._onCameraChangeEndFuncIndex);
                me._core.onPostDraw.start(me._onPostDrawFuncIndex);

            }

        };

        me.getNearestSurroundingsPossibleStep = function(steps) {
            var delta = me._step;
            var direction = me._step * (steps - me.getSteps()) > 0 ? 1 : -1;

            for(var adaptationSteps = me.getSteps(); adaptationSteps<me._core.getMaxStepsNumber(); adaptationSteps+=direction) {
                if(Math.abs(adaptationSteps - steps) <= delta) {
                    if(steps > adaptationSteps) {
                        return [adaptationSteps, adaptationSteps+me._step];

                    }

                    if(steps > adaptationSteps) {
                        return [adaptationSteps-me._step, adaptationSteps];

                    }

                    if(steps == adaptationSteps) {
                        return [adaptationSteps-me._step, adaptationSteps+me._step];

                    }
                }
            };

            return [me._core.getMaxStepsNumber()-me._step, me._core.getMaxStepsNumber()];
        };

        me.decreaseSteps = function() {
            var nearestSurroundingsPossibleSteps = me.getNearestSurroundingsPossibleStep(me._core.getSteps());
            me._steps = nearestSurroundingsPossibleSteps[0];
        };

        me.increaseSteps = function() {
            var nearestSurroundingsPossibleSteps = me.getNearestSurroundingsPossibleStep(me._core.getSteps());
            me._steps = nearestSurroundingsPossibleSteps[1];
        };

        me.getSteps = function() {
            return me._steps;
        };

        me.isRun = function() {
            var isRunOnPostDraw = me._core.onPostDraw.isStart(me._onPostDrawFuncIndex)
            var isRunOnCameraChangeStart = me._core.onCameraChangeStart.isStart(me._onCameraChangeEndFuncIndex);
            var isRunOnCameraChangeEnd = me._core.onCameraChangeEnd.isStart(me._onCameraChangeStartFuncIndex);

            return isRunOnPostDraw && isRunOnCameraChangeStart && isRunOnCameraChangeEnd;
        };

        me.isPause = function() {
            var isRunOnPostDraw = me._core.onPostDraw.isStart(me._onPostDrawFuncIndex)
            var isRunOnCameraChangeStart = me._core.onCameraChangeStart.isStart(me._onCameraChangeEndFuncIndex);
            var isRunOnCameraChangeEnd = me._core.onCameraChangeEnd.isStart(me._onCameraChangeStartFuncIndex);

            return !isRunOnPostDraw && !isRunOnCameraChangeStart && isRunOnCameraChangeEnd;
        };

        me._numberOfChanges = 0;

        me.do = function(fps) {

            if( fps < 10 && me.getSteps() > (me._step * 2) ) {
                me._numberOfChanges--;
                if(me._numberOfChanges == -5) {
                    me.decreaseSteps();
                    //console.log("FPS: " + fps + ", Number of steps: " + me.getSteps() );
                    me._numberOfChanges = 0;

                    me._core.setSteps( me.getSteps() );
                }


            } else if( fps > 30 && me.getSteps() < me._core.getMaxStepsNumber() ) {
                me._numberOfChanges++;
                if(me._numberOfChanges == 3) {
                    me.increaseSteps();
                    //console.log("FPS: " + fps + ", Number of steps: " + me.getSteps() );
                    me._numberOfChanges = 0;
                    me._core.setSteps( me.getSteps() );
                }

            }

        };

        return me;

    };

    namespace.AdaptationManager = AdaptationManager;

})(window.VRC);

(function(namespace) {
    var GeometryHelper = function() {

        var me = {};

        me.createBoxGeometry = function(geometryDimension, volumeSize, zFactor) {
            var vertexPositions = [
                //front face first
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]], 
                //front face second
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]], 

                // back face first
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                // back face second
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],

                // top face first
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                // top face second
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],

                // bottom face first
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                // bottom face second
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],

                // right face first
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                // right face second
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmax * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],

                // left face first
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                // left face second
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymin * volumeSize[1], geometryDimension.zmin * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmax * volumeSize[2]],
                [geometryDimension.xmin * volumeSize[0], geometryDimension.ymax * volumeSize[1], geometryDimension.zmin * volumeSize[2]]
            ];

            var vertexColors = [
                //front face first
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                //front face second
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmax],

                // back face first
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmin],
                // back face second
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmin],

                // top face first
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmin],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                // top face second
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmin],

                // bottom face first
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmax],
                // bottom face second
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmax],
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmax],

                // right face first
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                // right face second
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmax, geometryDimension.ymax, geometryDimension.zmax],
                [geometryDimension.xmax, geometryDimension.ymin, geometryDimension.zmax],

                // left face first
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmax],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmax],
                // left face second
                [geometryDimension.xmin, geometryDimension.ymin, geometryDimension.zmin],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmax],
                [geometryDimension.xmin, geometryDimension.ymax, geometryDimension.zmin]
            ];

            var positions = [];
            var colors = [];

            for(var i = 0; i < vertexPositions.length; i++) {
                var backCounter = vertexPositions.length - 1 - i;
                var x = vertexPositions[backCounter][0];
                var y = vertexPositions[backCounter][1];
                var z = vertexPositions[backCounter][2] * zFactor;

                var r = vertexColors[backCounter][0];
                var g = vertexColors[backCounter][1];
                var b = vertexColors[backCounter][2];

                positions.push(x);
                positions.push(y);
                positions.push(z);

                colors.push(r);
                colors.push(g);
                colors.push(b);
                colors.push(1.0);
            }

            var geometry = new THREE.BufferGeometry();
            var bufferPositions = new Float32Array(positions);
            geometry.addAttribute( 'position', new THREE.BufferAttribute( bufferPositions, 3 ) );
            geometry.addAttribute( 'vertColor', new THREE.BufferAttribute(new Float32Array(colors), 4));
            //geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            return geometry;
        }

        return me;
        
    };

    namespace.GeometryHelper = GeometryHelper;

})(window.VRC);

/**
 * @classdesc
 * Core
 *
 * @class Core
 * @this {Core}
 * @maintainer nicholas.jerome@kit.edu
 */

var Core = function(conf) {

    this.version = "1.0.0";

    // Zoom Box parameters
    this._zoom_parameters = {
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 1.0
    }

    // General Parameters
    this.zFactor = conf.zFactor != undefined ? conf.zFactor : 1;
    this._steps = conf.steps == undefined ? 20 : conf.steps;
    this._slices_gap = typeof conf.slices_range == undefined ? [0, '*'] : conf.slices_range;

    this._slicemap_row_col = [16, 16];
    this._gray_value = [0.0, 1.0];
    this._slicemaps_images = [];
    this._slicemaps_paths = conf.slicemaps_paths;
    this._slicemaps_width = [];
    this._slicemaps_textures = [];
    this._opacity_factor = conf.opacity_factor != undefined ? conf.opacity_factor : 35;
    this._color_factor = conf.color_factor != undefined ? conf.color_factor: 3;
    this._shader_name = conf.shader_name == undefined ? "secondPassDefault" : conf.shader_name;
    this._render_size = conf.renderer_size == undefined ? ['*', '*'] : conf.renderer_size;
    this._render_size_default = [256, 512];
    this._canvas_size = conf.renderer_canvas_size;
    this._render_clear_color = "#000";
    this._transfer_function_as_image = new Image();
    this._volume_sizes = [1024.0, 1024.0, 1024.0];
    this._geometry_dimensions = {
        "xmin": 0.0,
        "xmax": 1.0,
        "ymin": 0.0,
        "ymax": 1.0,
        "zmin": 0.0,
        "zmax": 1.0
    };

    this._transfer_function_colors = [
        {"color": "#000000", "pos": 0.0},
        {"color": "#ffffff", "pos": 1.0}
    ];

    this._dom_container_id = conf.dom_container != undefined ? conf.dom_container : "wave-container";
    this._dom_container = {};
    this._render = {};
    this._camera = {};
    this._camera_settings = {
        "rotation": {
            x: 0.0,
            y: 0.0,
            z: 0.0
        },
        "position": {
            "x": 0,
            "y": 0,
            "z": 3
        }
    };

    this._rtTexture = {};

    this._geometry = {};
    this._geometry_settings = {
        "rotation": {
            x: 0.0,
            y: 0.0,
            z: 0.0
        }
    };

    this._materialFirstPass = {};
    this._materialSecondPass = {};

    this._sceneFirstPass = {};
    this._sceneSecondPass = {};

    this._meshFirstPass = {};
    this._meshSecondPass = {};

    this.onPreDraw = new VRC.EventDispatcher();
    this.onPostDraw = new VRC.EventDispatcher();
    this.onResizeWindow = new VRC.EventDispatcher();
    this.onCameraChange = new VRC.EventDispatcher();
    this.onCameraChangeStart = new VRC.EventDispatcher();
    this.onCameraChangeEnd = new VRC.EventDispatcher();
    this.onChangeTransferFunction = new VRC.EventDispatcher();

    this._onWindowResizeFuncIndex_canvasSize = -1;
    this._onWindowResizeFuncIndex_renderSize = -1;

    this._callback = conf.callback;

    try {
        if(this._canvas_size[0] > this._canvas_size[1])
            this._camera_settings.position.z = 2;
    } catch(e){}
};

Core.prototype.init = function() {
    var me = this;

    this._container = this.getDOMContainer();

    this._render = new THREE.WebGLRenderer({
        preserveDrawingBuffer: true,
        antialias: true,
        alpha : true
    });
    this._render.domElement.id = 'wave-'+this._dom_container_id;
    this._render.setSize(this.getRenderSizeInPixels()[0],
                         this.getRenderSizeInPixels()[1]);
    this._render.setClearColor(this._render_clear_color, 0);

    // this._container.removeChild( this._container.firstChild );
    this._container.innerHTML=" ";

    this._container.appendChild( this._render.domElement );

    this._camera = new THREE.PerspectiveCamera(
        45,
        this.getRenderSizeInPixels()[0] / this.getRenderSizeInPixels()[1],
        0.01,
        11
    );
    this._camera.position.x = this._camera_settings["position"]["x"];
    this._camera.position.y = this._camera_settings["position"]["y"];
    this._camera.position.z = this._camera_settings["position"]["z"];

    this._camera.rotation.x = this._camera_settings["rotation"]["x"];
    this._camera.rotation.y = this._camera_settings["rotation"]["y"];
    this._camera.rotation.z = this._camera_settings["rotation"]["z"];

    this.isAxisOn = false;

    // Control
    this._controls = new THREE.TrackballControls(
        this._camera,
        this._render.domElement);
    this._controls.rotateSpeed = 2.0;
    this._controls.zoomSpeed = 2.0;
    this._controls.panSpeed = 2.0;

    this._controls.noZoom = false;
    this._controls.noPan = true;

    this._controls.staticMoving = true;
    this._controls.dynamicDampingFactor = 0.1;

    this._controls.autoRotate = true;


    this._rtTexture = new THREE.WebGLRenderTarget(
        this.getRenderSizeInPixels()[0],
        this.getRenderSizeInPixels()[1],
        {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            wrapS:  THREE.ClampToEdgeWrapping,
            wrapT:  THREE.ClampToEdgeWrapping,
            format: THREE.RGBFormat,
            type: THREE.UnsignedByteType,
            generateMipmaps: false
        }
    );

        this._materialFirstPass = new THREE.ShaderMaterial( {
        vertexShader: this._shaders.firstPass.vertexShader,
        fragmentShader: this._shaders.firstPass.fragmentShader,
        side: THREE.FrontSide,
        //side: THREE.BackSide,
        transparent: true
        } );

        // TODO: Load colourmap, but should be from local
        //var cm = THREE.ImageUtils.loadTexture( "http://katrin.kit.edu/vis/colormap/cm_jet.png" );
        //cm.minFilter = THREE.LinearFilter;
        this._materialSecondPass = new THREE.ShaderMaterial( {
        vertexShader: this._shaders[this._shader_name].vertexShader,
        fragmentShader: ejs.render( this._shaders[this._shader_name].fragmentShader, {
            "maxTexturesNumber": me.getMaxTexturesNumber()}),
        uniforms: {
            uRatio : { type: "f", value: this.zFactor},
            uBackCoord: { type: "t",  value: this._rtTexture },
            uSliceMaps: { type: "tv", value: this._slicemaps_textures },
            uLightPos: {type:"v3", value: new THREE.Vector3() },
            uSetViewMode: {type:"i", value: 0 },
            //uColormap : {type:'t',value:cm },
            uSteps: { type: "i", value: this._steps },
            uSlicemapWidth: { type: "f", value: this._slicemaps_width },
            uNumberOfSlices: { type: "f", value: parseFloat(this.getSlicesRange()[1]) },
            uSlicesOverX: { type: "f", value: this._slicemap_row_col[0] },
            uSlicesOverY: { type: "f", value: this._slicemap_row_col[1] },
            uOpacityVal: { type: "f", value: this._opacity_factor },
            darkness: { type: "f", value: this._color_factor },


            uTransferFunction: { type: "t",  value: this._transfer_function },
            uColorVal: { type: "f", value: this._color_factor },
            uAbsorptionModeIndex: { type: "f", value: this._absorption_mode_index },
            uMinGrayVal: { type: "f", value: this._gray_value[0] },
            uMaxGrayVal: { type: "f", value: this._gray_value[1] },
            uIndexOfImage: { type: "f", value: this._indexOfImage }
        },
        //side: THREE.FrontSide,
        side: THREE.BackSide,
        transparent: true
        });

        this._sceneFirstPass = new THREE.Scene();
        this._sceneSecondPass = new THREE.Scene();

        // Created mesh for both passes using geometry helper
        this._initGeometry( this.getGeometryDimensions(), this.getVolumeSizeNormalized() );
        this._meshFirstPass = new THREE.Mesh( this._geometry, this._materialFirstPass );
        this._meshSecondPass = new THREE.Mesh( this._geometry, this._materialSecondPass );

        //this._axes = buildAxes(0.5);
        this._sceneFirstPass.add(this._meshFirstPass);
        this._sceneSecondPass.add(this._meshSecondPass);
        //this._sceneSecondPass.add(this._axes);

        var mesh = new THREE.Mesh(
            new THREE.BoxGeometry( 1, 1, 1 ),
            new THREE.MeshNormalMaterial()
        );
        this._wireframe = new THREE.BoxHelper( mesh );
        this._wireframe.material.color.set( 0xe3e3e3 );
        this._sceneSecondPass.add( this._wireframe );

        var mesh_zoom = new THREE.Mesh(
            new THREE.BoxGeometry( 1.0, 1.0, 1.0 ),
            new THREE.MeshNormalMaterial()
        );
        this._wireframe_zoom = new THREE.BoxHelper( mesh_zoom );
        this._wireframe_zoom.material.color.set( 0x0000ff );
        this._sceneSecondPass.add( this._wireframe_zoom );
        
        var sphere = new THREE.SphereGeometry( 0.1 );
        this._light1 = new THREE.PointLight( 0xff0040, 2, 50 );
        this._light1.add( new THREE.Mesh( sphere, new THREE.MeshBasicMaterial( { color: 0xff0040 } ) ) );
        this._light1.position.set(1, 0, 0);
        //this._sceneSecondPass.add( this._light1 );
        //this._sceneSecondPass.add( new THREE.DirectionalLightHelper(this._light1, 0.2) );

        // parent
        this._parent = new THREE.Object3D();
        this._sceneSecondPass.add( this._parent );
        // pivot
        this._pivot = new THREE.Object3D();
        this._parent.add( this._pivot );


    this.setTransferFunctionByColors(this._transfer_function_colors);
    this.setGeometryDimensions(this.getGeometryDimensions());


    window.addEventListener( 'resize', function() {
        //console.log("WAVE: trigger: resize");
        me.onResizeWindow.call();
    }, false );

    this._controls.addEventListener("change", function() {
        //console.log("WAVE: trigger: change");
        me.onCameraChange.call();
    });

    this._controls.addEventListener("scroll", function() {
        //console.log("WAVE: trigger: scroll");
        me.onCameraChange.call();
    });

    this._controls.addEventListener("start", function() {
        //console.log("WAVE: start trigger");
        me.onCameraChangeStart.call();
    });

    this._controls.addEventListener("end", function() {
        //console.log("WAVE: end trigger");
        me.onCameraChangeEnd.call();
    });

    this._onWindowResizeFuncIndex_canvasSize = this.onResizeWindow.add(function() {
        me.setRenderCanvasSize('*', '*');
    }, false);

    this._render.setSize(this.getRenderSizeInPixels()[0], this.getRenderSizeInPixels()[1]);
    this.setRenderCanvasSize(this.getCanvasSize()[0], this.getCanvasSize()[1]);

    try{
        this._callback();
    } catch(e){}
};


/**
 * API 
 *
 **/
Core.prototype.getVersion = function() {
    console.log(this.version);
};

Core.prototype.getRenderSizeDefault = function() {
    return this._render_size_default;
};

Core.prototype.setRenderSizeDefault = function(value) {
    this._render_size_default = [value[0], value[1]];
    this._render_size = [value[1], value[1]];
    this._render.setSize(this.getRenderSizeInPixels()[0], this.getRenderSizeInPixels()[1]);
    this.setRenderCanvasSize(this.getCanvasSize()[0], this.getCanvasSize()[1]);
    console.log(this._render_size);
};

Core.prototype.setRenderSize = function(value) {
    this._render_size = [value, value];
    this._render.setSize(this.getRenderSizeInPixels()[0], this.getRenderSizeInPixels()[1]);
    this.setRenderCanvasSize(this.getCanvasSize()[0], this.getCanvasSize()[1]);
    console.log(this._render_size);
};

Core.prototype._setUpBox = function(parameters) {
    width = parameters.xmax - parameters.xmin;
    height = parameters.ymax - parameters.ymin;
    depth = parameters.zmax - parameters.zmin;
    this._wireframe_zoom.scale.x = width;
    this._wireframe_zoom.scale.y = height;
    this._wireframe_zoom.scale.z = depth;
    this._wireframe_zoom.position.x = (parameters.xmax - 0.5) - (width / 2.0 );
    this._wireframe_zoom.position.y = (parameters.ymax - 0.5) - (height / 2.0 );
    this._wireframe_zoom.position.z = (parameters.zmax - 0.5) - (depth / 2.0 );
};


Core.prototype.setZoomColor = function(value) {
    this._wireframe_zoom.material.color.set( value );
};


Core.prototype.setZoomXMinValue = function(value) {
    this._zoom_parameters.xmin = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.setZoomXMaxValue = function(value) {
    this._zoom_parameters.xmax = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.setZoomYMinValue = function(value) {
    this._zoom_parameters.ymin = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.setZoomYMaxValue = function(value) {
    this._zoom_parameters.ymax = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.setZoomZMinValue = function(value) {
    this._zoom_parameters.zmin = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.setZoomZMaxValue = function(value) {
    this._zoom_parameters.zmax = value;
    this._setUpBox( this._zoom_parameters );
};


Core.prototype.showZoomBox = function(value) {
    if (value == true) {
        this._sceneSecondPass.add( this._wireframe_zoom );
    } else {
        this._sceneSecondPass.remove( this._wireframe_zoom );
    }
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype._secondPassSetUniformValue = function(key, value) {
    this._materialSecondPass.uniforms[key].value = value;
};


Core.prototype._setSlicemapsTextures = function(imagePaths) {
    var allPromises = [];
    var me = this;
    var textures = [];
    var loader = new THREE.TextureLoader();
    loader.crossOrigin = ''; 
    
    imagePaths.forEach( function( path ) {
        allPromises.push( new Promise( function( resolve, reject ) {

            loader.load(path, function (texture) {
                texture.magFilter = THREE.LinearFilter;
                texture.minFilter = THREE.LinearFilter;
                texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                texture.generateMipmaps = false;
                texture.flipY = false;
                texture.needsUpdate = true;
                //textures.push(texture);
                resolve( texture );
            }, 
            function( xhr ) {
               // Progress callback of TextureLoader
               // ...
            },    
            function (err) {
                console.log(err);
                console.log("error");
            });
        }));
    });
    Promise.all( allPromises )
        .then( function( promises ) {
            // All textures are now loaded, and this array
            // contains all the materials that you created
            me._secondPassSetUniformValue("uSliceMaps", promises);
            this._slicemaps_textures = promises;
            this._slicemaps_width = promises[0].image.width;
            me._secondPassSetUniformValue("uSlicemapWidth", this._slicemaps_width);
        }, function( error ) {
            console.error( "Could not load all textures:", error );
        });
};


Core.prototype.setTransferFunctionByImage = function(image) {
    this._transfer_function_as_image = image;
    var texture = new THREE.Texture(image);
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.wrapS = texture.wrapT =  THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.flipY = true;
    texture.needsUpdate = true;
    this._secondPassSetUniformValue("uTransferFunction", texture);
    //this.onChangeTransferFunction.call(image);
};


Core.prototype.setTransferFunctionByColors = function(colors) {
    this._transfer_function_colors = colors;
    var canvas = document.createElement('canvas');
    canvas.width  = 512;
    canvas.height = 2;
    var ctx = canvas.getContext('2d');

    var grd = ctx.createLinearGradient(0, 0, canvas.width -1 , canvas.height - 1);

    for(var i=0; i<colors.length; i++) {
        grd.addColorStop(colors[i].pos, colors[i].color);
    }

    ctx.fillStyle = grd;
    ctx.fillRect(0,0,canvas.width ,canvas.height);
    var image = new Image();
    image.src = canvas.toDataURL();
    image.style.width = 20 + " px";
    image.style.height = 512 + " px";
    
    var transferTexture = this.setTransferFunctionByImage(image);

    this.onChangeTransferFunction.call(image);
};


Core.prototype.getTransferFunctionAsImage = function() {
    return this._transfer_function_as_image;
};


Core.prototype.getBase64 = function() {
    return this._render.domElement.toDataURL("image/jpeg");
};


Core.prototype._initGeometry = function(geometryDimensions, volumeSizes) {
    var geometryHelper = new VRC.GeometryHelper();
    this._geometry = geometryHelper.createBoxGeometry(geometryDimensions, volumeSizes, this.zFactor);

    this._geometry.applyMatrix( new THREE.Matrix4().makeTranslation( -volumeSizes[0] / 2, -volumeSizes[1] / 2, -volumeSizes[2] / 2 ) );
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationX( this._geometry_settings["rotation"]["x"] ));
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationY( this._geometry_settings["rotation"]["y"] ));
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationZ( this._geometry_settings["rotation"]["z"] ));
    this._geometry.doubleSided = true;
};


Core.prototype.setMode = function(conf) {
    this._shader_name =  conf.shader_name;


        this._materialSecondPass = new THREE.ShaderMaterial( {
            vertexShader: this._shaders[this._shader_name].vertexShader,
            fragmentShader: ejs.render(
                this._shaders[this._shader_name].fragmentShader,
                {"maxTexturesNumber": this.getMaxTexturesNumber()}
            ),
            //attributes: {
            //    vertColor: {type: 'c', value: [] }
            //},
            uniforms: {
                uRatio : { type: "f", value: this.zFactor},
                uBackCoord: { type: "t",  value: this._rtTexture },
                uSliceMaps: { type: "tv", value: this._slicemaps_textures },
                uLightPos: {type:"v3", value: new THREE.Vector3() },
                uSetViewMode: {type:"i", value: 0 },
                uNumberOfSlices: { type: "f", value: parseFloat(this.getSlicesRange()[1]) },
                uSlicemapWidth: { type: "f", value: this._slicemaps_width},
                uSlicesOverX: { type: "f", value: this._slicemap_row_col[0] },
                uSlicesOverY: { type: "f", value: this._slicemap_row_col[1] },
                uOpacityVal: { type: "f", value: this._opacity_factor },
            },
            side: THREE.BackSide,
            transparent: true
        });

        this._meshSecondPass = new THREE.Mesh( this._geometry, this._materialSecondPass );

        this._sceneSecondPass = new THREE.Scene();
        this._sceneSecondPass.add( this._meshSecondPass );
}


Core.prototype.setZoom = function(x1, x2, y1, y2) {
    //this._material2D.uniforms["uZoom"].value = new THREE.Vector4(0.1875, 0.28125, 0.20117, 0.29492);
    this._material2D.uniforms["uZoom"].value = new THREE.Vector4(x1, x2, y1, y2);
    //uSetViewMode: {type: "i", value: 0 }
    //this._material2D.uniforms.uZoom.value = {type: "i", value: 1 };
}

Core.prototype.set2DTexture = function(urls) {
    var chosen_cm = THREE.ImageUtils.loadTexture( urls[0] );
    var chosen_cm2 = THREE.ImageUtils.loadTexture( urls[1] );

    chosen_cm.minFilter = THREE.NearestFilter;
    chosen_cm2.minFilter = THREE.NearestFilter;

    this._material2D.uniforms["texture1"].value = chosen_cm ;
    this._material2D.uniforms["texture2"].value = chosen_cm2;
    this._material2D.needsUpdate = true;
}

/////////////////////////////////////////////////////////////////////
Core.prototype.setShaderName = function(value) {

    // new THREE.BoxGeometry( 1, 1, 1 ),

    this._shader_name = value;
    // this._shader_name =  conf.shader_name;

        this._materialSecondPass = new THREE.ShaderMaterial( {
            vertexShader: this._shaders[this._shader_name].vertexShader,
            fragmentShader: ejs.render( this._shaders[this._shader_name].fragmentShader, {
                "maxTexturesNumber": this.getMaxTexturesNumber()
            }),
            uniforms: {
                uRatio : { type: "f", value: this.zFactor},
                uBackCoord: { type: "t",  value: this._rtTexture },
                uSliceMaps: { type: "tv", value: this._slicemaps_textures },
                uLightPos: {type:"v3", value: new THREE.Vector3() },
                uSetViewMode: {type:"i", value: 0 },

                uSteps: { type: "i", value: this._steps },
                uSlicemapWidth: { type: "f", value: this._slicemaps_width },
                uNumberOfSlices: { type: "f", value: parseFloat(this.getSlicesRange()[1]) },
                uSlicesOverX: { type: "f", value: this._slicemap_row_col[0] },
                uSlicesOverY: { type: "f", value: this._slicemap_row_col[1] },
                uOpacityVal: { type: "f", value: this._opacity_factor },

                uTransferFunction: { type: "t",  value: this._transfer_function },
                uColorVal: { type: "f", value: this._color_factor },
                uAbsorptionModeIndex: { type: "f", value: this._absorption_mode_index },
                uMinGrayVal: { type: "f", value: this._gray_value[0] },
                uMaxGrayVal: { type: "f", value: this._gray_value[1] },
                uIndexOfImage: { type: "f", value: this._indexOfImage },

                uSosThresholdBot: { type: "f", value: this._sosThresholdBot },
                uSosThresholdTop: { type: "f", value: this._sosThresholdTop },
                uAttenThresholdBot: { type: "f", value: this._attenThresholdBot },
                uAttenThresholdTop: { type: "f", value: this._attenThresholdTop },
            },
            //side: THREE.FrontSide,
            side: THREE.BackSide,
            transparent: true
        });


        this._meshSecondPass = new THREE.Mesh( this._geometry, this._materialSecondPass );

        this._sceneSecondPass = new THREE.Scene();
        this._sceneSecondPass.add( this._meshSecondPass );

        this.addWireframe();

}
/////////////////////////////////////////////////////////////////////



Core.prototype.setShader = function(codeblock) {
    var header = "uniform vec2 resolution; \
    precision mediump int; \
    precision mediump float; \
    varying vec4 pos; \
    uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>]; \
    uniform sampler2D texture1; \
    uniform sampler2D texture2; \
    uniform sampler2D colourmap; \
    uniform vec4 uZoom; \
    void main(void) { \
    vec2 pos = gl_FragCoord.xy / resolution.xy; \
    float b1, b2, b3, b4, b5, b6; \
    vec3 t1, t2; \
    float newX = ((uZoom.y - uZoom.x)  * pos.x) + uZoom.x; \
    float newY = ((uZoom.w - uZoom.z)  * pos.y) + uZoom.z; \
    t1 = texture2D(texture1, vec2(newX, newY)).xyz; \
    t2 = texture2D(texture2, vec2(newX, newY)).xyz; \
    b1 = t1.x; \
    b2 = t1.y; \
    b3 = t1.z; \
    b4 = t2.x; \
    b5 = t2.y; \
    b6 = t2.z;";
    var footer = "}";

    var final_code = header + codeblock + footer;

    this._sceneFirstPass.remove(this._meshFirstPass);
    this._material2D = new THREE.ShaderMaterial({
        vertexShader: this._shaders["secondPass2DCustom"].vertexShader,
        fragmentShader: ejs.render(
            final_code,
            {"maxTexturesNumber": this.getMaxTexturesNumber()}
        ),
        uniforms: {
            uSetViewMode: {type: "i", value: 0 },
            texture1: {type: 't', value: this._tex1},
            texture2: {type: 't', value: this._tex2},
            colourmap: {type: 't', value: this._cm},
            uZoom: {type:'v4', value: new THREE.Vector4(0.0, 1.0, 0.0, 1.0)},
            resolution: {type: 'v2',value: new THREE.Vector2(this._render_size[0], this._render_size[1])}
        }
    });
    var geometry = new THREE.PlaneBufferGeometry( 10, 10 );
    this._meshFirstPass = new THREE.Mesh( geometry, this._material2D );
    this._sceneFirstPass = new THREE.Scene();
    this._sceneFirstPass.add(this._meshFirstPass);
}


Core.prototype._setGeometry = function(geometryDimensions, volumeSizes) {
    var geometryHelper = new VRC.GeometryHelper();
    var geometry      = geometryHelper.createBoxGeometry(geometryDimensions, volumeSizes, 1.0);
    //var geometry      = geometryHelper.createBoxGeometry(geometryDimensions, volumeSizes, this.zFactor);
    var colorArray    = geometry.attributes.vertColor.array;
    var positionArray = geometry.attributes.position.array;

    this._geometry.attributes.vertColor.array = colorArray;
    this._geometry.attributes.vertColor.needsUpdate = true;

    this._geometry.attributes.position.array = positionArray;
    this._geometry.attributes.position.needsUpdate = true;

    this._geometry.applyMatrix( new THREE.Matrix4().makeTranslation( -volumeSizes[0] / 2, -volumeSizes[1] / 2, -volumeSizes[2] / 2 ) );
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationX( this._geometry_settings["rotation"]["x"] ));
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationY( this._geometry_settings["rotation"]["y"] ));
    this._geometry.applyMatrix( new THREE.Matrix4().makeRotationZ( this._geometry_settings["rotation"]["z"] ));

    this._geometry.doubleSided = true;
};


Core.prototype.setSlicemapsImages = function(images, imagesPaths) {
    this._slicemaps_images = images;
    this._slicemaps_paths = imagesPaths != undefined ? imagesPaths : this._slicemaps_paths;
    this._setSlicemapsTextures(images);
    this._secondPassSetUniformValue("uSliceMaps", this._slicemaps_textures);
};


Core.prototype.setSteps = function(steps) {
    this._steps = steps;
    this._secondPassSetUniformValue("uSteps", this._steps);
};


Core.prototype.setSlicesRange = function(from, to) {
    this._slices_gap = [from, to];
    this._secondPassSetUniformValue("uNumberOfSlices", parseFloat(this.getSlicesRange()[1]));
};


Core.prototype.setOpacityFactor = function(opacity_factor) {
    this._opacity_factor = opacity_factor;
    this._secondPassSetUniformValue("uOpacityVal", this._opacity_factor);
};


Core.prototype.setColorFactor = function(color_factor) {
    this._color_factor = color_factor;
    this._secondPassSetUniformValue("darkness", this._color_factor);
};


Core.prototype.setAbsorptionMode = function(mode_index) {
    this._absorption_mode_index = mode_index;
    this._secondPassSetUniformValue("uAbsorptionModeIndex", this._absorption_mode_index);
};

Core.prototype.setIndexOfImage = function(indexOfImage) {
    this._indexOfImage = indexOfImage;
    this._secondPassSetUniformValue("uIndexOfImage", this._indexOfImage);
};


Core.prototype.setVolumeSize = function(width, height, depth) {
    this._volume_sizes = [width, height, depth];

    var maxSize = Math.max(this.getVolumeSize()[0], this.getVolumeSize()[1], this.getVolumeSize()[2]);
    var normalizedVolumeSizes = [this.getVolumeSize()[0] / maxSize,  this.getVolumeSize()[1] / maxSize, this.getVolumeSize()[2] / maxSize];

    this._setGeometry(this.getGeometryDimensions(), normalizedVolumeSizes);
};


Core.prototype.setGeometryDimensions = function(geometryDimension) {
    this._geometry_dimensions = geometryDimension;

    this._setGeometry(this._geometry_dimensions, this.getVolumeSizeNormalized());
};


Core.prototype.setRenderCanvasSize = function(width, height) {
    this._canvas_size = [width, height];

    if( (this._canvas_size[0] == '*' || this._canvas_size[1] == '*') && !this.onResizeWindow.isStart(this._onWindowResizeFuncIndex_canvasSize) ) {
        this.onResizeWindow.start(this._onWindowResizeFuncIndex_canvasSize);
    }

    if( (this._canvas_size[0] != '*' || this._canvas_size[1] != '*') && this.onResizeWindow.isStart(this._onWindowResizeFuncIndex_canvasSize) ) {
        this.onResizeWindow.stop(this._onWindowResizeFuncIndex_canvasSize);
    }

    var width = this.getCanvasSizeInPixels()[0];
    var height = this.getCanvasSizeInPixels()[1];

    this._render.domElement.style.width = width + "px";
    this._render.domElement.style.height = height + "px";

    this._camera.aspect = width / height;
    this._camera.updateProjectionMatrix();
};


Core.prototype.setBackgroundColor = function(color) {
    this._render_clear_color = color;
    this._render.setClearColor(color);
};


Core.prototype.setRowCol = function(row, col) {
    this._slicemap_row_col = [row, col];
    this._secondPassSetUniformValue("uSlicesOverX", this._slicemap_row_col[0]);
    this._secondPassSetUniformValue("uSlicesOverY", this._slicemap_row_col[1]);
};


Core.prototype.setGrayMinValue = function(value) {
    this._gray_value[0] = value;
    this._secondPassSetUniformValue("uMinGrayVal", this._gray_value[0]);
};


Core.prototype.applyThresholding = function(threshold_name) {
    switch( threshold_name ) {
        case "otsu": {
            this.setGrayMinValue( this._threshold_otsu_index );
        }; break;

        case "isodata": {
            this.setGrayMinValue( this._threshold_isodata_index );
        }; break;

        case "yen": {
            this.setGrayMinValue( this._threshold_yen_index );
        }; break;

        case "li": {
            this.setGrayMinValue( this._threshold_li_index );
        }; break;
    }
};


Core.prototype.setThresholdIndexes = function(otsu, isodata, yen, li) {
    this._threshold_otsu_index       = otsu;
    this._threshold_isodata_index    = isodata;
    this._threshold_yen_index        = yen;
    this._threshold_li_index         = li;
};


Core.prototype.setGrayMaxValue = function(value) {
    this._gray_value[1] = value;
    this._secondPassSetUniformValue("uMaxGrayVal", this._gray_value[1]);
};


Core.prototype.startRotate = function() {
    this._isRotate = true;
};


Core.prototype.stopRotate = function() {
    this._isRotate = false;
};


Core.prototype.addWireframe = function() {
    this._sceneSecondPass.add( this._wireframe );
    this._render.render( this._sceneFirstPass, this._camera, this._rtTexture, true );
    this._render.render( this._sceneFirstPass, this._camera );

    // Render the second pass and perform the volume rendering.
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype.removeWireframe = function() {
    this._sceneSecondPass.remove( this._wireframe );
    this._render.render( this._sceneFirstPass, this._camera, this._rtTexture, true );
    this._render.render( this._sceneFirstPass, this._camera );

    // Render the second pass and perform the volume rendering.
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype.setAxis = function(value) {
    if (this.isAxisOn) {
        this._sceneSecondPass.remove(this._axes);
        this.isAxisOn = false;
    } else {
        this._sceneSecondPass.add(this._axes);
        this.isAxisOn = true;
    }

    this._render.render( this._sceneFirstPass, this._camera, this._rtTexture, true );
    this._render.render( this._sceneFirstPass, this._camera );

    // Render the second pass and perform the volume rendering.
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype.showISO = function() {
    this._secondPassSetUniformValue("uSetViewMode", 1);
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype.showVolren = function() {
    this._secondPassSetUniformValue("uSetViewMode", 0);
    this._render.render( this._sceneSecondPass, this._camera );
};


Core.prototype.draw = function(fps) {
    this.onPreDraw.call(fps.toFixed(3));

    var cameraPosition = this._light1.getWorldPosition();
    this._secondPassSetUniformValue("uLightPos", cameraPosition);

    this._controls.update();

    this._render.render( this._sceneFirstPass, this._camera, this._rtTexture, true );
    this._render.render( this._sceneFirstPass, this._camera );

    // Render the second pass and perform the volume rendering.
    this._render.render( this._sceneSecondPass, this._camera );
    this.onPostDraw.call(fps.toFixed(3));
};


Core.prototype.getDOMContainer = function() {
    return document.getElementById(this._dom_container_id);
};


Core.prototype.getRenderSize  = function() {
    var width = this._render_size[0];
    var height = this._render_size[1];

    return [width, height];
};


Core.prototype.getRenderSizeInPixels  = function() {
    var width = this.getRenderSize()[0];
    var height = this.getRenderSize()[0];

    if(this._render_size[0] == '*') {
        width = this.getCanvasSizeInPixels()[0];
    }
    if(this._render_size[1] == '*') {
        height = this.getCanvasSizeInPixels()[1];
    }

    return [width, height];
};


Core.prototype.getCanvasSize = function() {
    var width = this._canvas_size[0];
    var height = this._canvas_size[1];

    return [width, height];
};


Core.prototype.getCanvasSizeInPixels = function() {
    var width = this.getCanvasSize()[0];
    var height = this.getCanvasSize()[1];
    var canvas_id = "#" + this._dom_container_id + " > canvas";
    var container = document.getElementById(this._dom_container_id);

    if(this._canvas_size[0] == '*') {
        width = document.querySelector(canvas_id).width;
        container.style.width = width+"px";
    } else if (this._canvas_size[0] == 'fullscreen') {
        width = window.innerWidth
        || document.documentElement.clientWidth
        || document.body.clientWidth;
        container.style.width = width+"px";
    }

    if(this._canvas_size[1] == '*') {
        height = document.querySelector(canvas_id).height;
        container.style.height = height+"px";
    } else if (this._canvas_size[1] == 'fullscreen') {
        height = window.innerHeight
        || document.documentElement.clientHeight
        || document.body.clientHeight;
        container.style.height = height+"px";
    }
    return [width, height];
};


Core.prototype.getSteps = function() {
    return this._steps;
};


Core.prototype.getSlicemapsImages = function() {
    return this._slicemaps_images;
};


Core.prototype.getSlicemapsPaths = function() {
    return this._slicemaps_paths;
};


Core.prototype.getRowCol = function() {
    return this._slicemap_row_col;
};


Core.prototype.getSlicesRange  = function() {
    var from = this._slices_gap[0];
    var to = this._slices_gap[1];
    if(this._slices_gap[1] == '*') {
        to = (this.getRowCol()[0] * this.getRowCol()[1] * this.getSlicemapsImages().length) - 1;
    }

    return [from, to];
};


Core.prototype.getVolumeSize = function() {
    return this._volume_sizes;
};


Core.prototype.getMaxStepsNumber = function() {
    return Math.min( this.getVolumeSize()[0], this.getVolumeSize()[1] );
};


Core.prototype.getVolumeSizeNormalized = function() {
    var maxSize = Math.max(this.getVolumeSize()[0],
                           this.getVolumeSize()[1],
                           this.getVolumeSize()[2]);
    var normalizedVolumeSizes = [
        parseFloat(this.getVolumeSize()[0]) / parseFloat(maxSize),
        parseFloat(this.getVolumeSize()[1]) / parseFloat(maxSize),
        parseFloat(this.getVolumeSize()[2]) / parseFloat(maxSize)];

    return normalizedVolumeSizes;
};


Core.prototype.getGeometryDimensions = function() {
    return this._geometry_dimensions;

};


Core.prototype.getGrayMinValue = function() {
    return this._gray_value[0];
};


Core.prototype.getGrayMaxValue = function() {
    return this._gray_value[1];
};


Core.prototype.getClearColor = function() {
    return this._render_clear_color;
};


Core.prototype.getTransferFunctionColors = function() {
    return this._transfer_function_colors;
};


Core.prototype.getOpacityFactor = function() {
    return this._opacity_factor;
};


Core.prototype.getColorFactor = function() {
    return this._color_factor;
};


Core.prototype.getAbsorptionMode = function() {
    return this._absorption_mode_index;
};

// Core.prototype.getIndexOfImage = function() {
//     return this._indexOfImage;
// };


Core.prototype.getClearColor = function() {
    return this._render_clear_color;
};


Core.prototype.getDomContainerId = function() {
    return this._dom_container_id;
};


Core.prototype.getMaxTexturesNumber = function() {
    var number_used_textures = 6;
    var gl = this._render.getContext()
    return gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) - number_used_textures;
};


Core.prototype.getMaxTextureSize = function() {
    var gl = this._render.getContext()
    return gl.getParameter(gl.MAX_TEXTURE_SIZE);
};


Core.prototype.getMaxFramebuferSize = function() {
    var gl = this._render.getContext()
    return gl.getParameter(gl.MAX_RENDERBUFFER_SIZE);
};


Core.prototype._shaders = {
    // Here will be inserted shaders withhelp of grunt
};


function buildAxes( length ) {
    var axes = new THREE.Object3D();

    // This is just intended as a building block for drawing axis
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( (length*3.0), -length, -length ), 0xFF0000, false ) ); // +X //red
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( (-length*2.0), -length, -length ), 0xFF0000, true ) ); // -X
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( -length, (length*3.0), -length ), 0x00FF00, false ) ); // +Y //green
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( -length, (-length*2.0), -length ), 0x00FF00, true ) ); // -Y
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( -length, -length, (length*3.0) ), 0x0000FF, false ) ); // +Z //blue
    axes.add( buildAxis( new THREE.Vector3( -length, -length, -length ), new THREE.Vector3( -length, -length, (-length*2.0) ), 0x0000FF, true ) ); // -Z

    return axes;
}

function buildAxis( src, dst, colorHex, dashed ) {
    var geom = new THREE.Geometry(),
	mat;

    if(dashed) {
        mat = new THREE.LineDashedMaterial({ linewidth: 1, color: colorHex, dashSize: 3, gapSize: 3 });
    } else {
        mat = new THREE.LineBasicMaterial({ linewidth: 1, color: colorHex });
    }

    geom.vertices.push( src.clone() );
    geom.vertices.push( dst.clone() );
    geom.computeLineDistances(); // This one is SUPER important, otherwise dashed lines will appear as simple plain lines

    var axis = new THREE.Line( geom, mat, THREE.LinePieces );

    return axis;
}


window.VRC.Core = Core;

window.VRC.Core.prototype._shaders.firstPass = {
	uniforms: THREE.UniformsUtils.merge([
		{
		}
	]),
	vertexShader: [
		'varying vec3 worldSpaceCoords;',
		'void main()',
		'{',
		'    //Set the world space coordinates of the back faces vertices as output.',
		'    worldSpaceCoords = position + vec3(0.5, 0.5, 0.5); //move it from [-0.5;0.5] to [0,1]',
		'    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
		'}  '].join("\n"),
	fragmentShader: [
		'varying vec3 worldSpaceCoords;',
		'void main() {',
		'    //The fragment\'s world space coordinates as fragment output.',
		'    gl_FragColor = vec4( worldSpaceCoords.x , worldSpaceCoords.y, worldSpaceCoords.z, 1 );',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPass = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSteps" : { type: "f", value: -1 },
		"uAvailable_textures_number" : { type: "i", value: 0 },
		}
	]),
	vertexShader: [
		'precision mediump int; ',
		'precision mediump float; ',
		'attribute vec4 vertColor; ',
		'varying vec4 frontColor; ',
		'varying vec4 pos; ',
		'void main(void) ',
		'{ ',
		'    frontColor = vertColor; ',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0); ',
		'    gl_Position = pos; ',
		'} '].join("\n"),
	fragmentShader: [
		'/**',
		'Usage:',
		'    1. Set mode:',
		'        rcl2.setAbsorptionMode(1.0) ',
		'    2. Set threshold:',
		'        rcl2.setGrayMinValue(0.43)',
		' **/',
		'#ifdef GL_FRAGMENT_PRECISION_HIGH ',
		' // highp is supported ',
		' precision highp int; ',
		' precision highp float; ',
		'#else ',
		' // high is not supported ',
		' precision mediump int; ',
		' precision mediump float; ',
		'#endif ',
		'varying vec4 frontColor; ',
		'varying vec4 pos; ',
		'uniform sampler2D uBackCoord; ',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uNumberOfSlices; ',
		'uniform float uMinGrayVal; ',
		'uniform float uMaxGrayVal; ',
		'uniform float uOpacityVal; ',
		'uniform float uColorVal; ',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX; ',
		'uniform float uSlicesOverY; ',
		'uniform float uSteps; ',
		'// uniform int uAvailable_textures_number;',
		'//Acts like a texture3D using Z slices and trilinear filtering. ',
		'vec4 getVolumeValue(vec3 volpos)',
		'{',
		'    float s1Original, s2Original, s1, s2; ',
		'    float dx1, dy1; ',
		'    // float dx2, dy2; ',
		'    // float value; ',
		'    vec2 texpos1,texpos2; ',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY; ',
		'    s1Original = floor(volpos.z*uNumberOfSlices); ',
		'    // s2Original = min(s1Original + 1.0, uNumberOfSlices);',
		'    int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'    // int tex2Index = int(floor(s2Original / slicesPerSprite));',
		'    s1 = mod(s1Original, slicesPerSprite);',
		'    // s2 = mod(s2Original, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    // dx2 = fract(s2/uSlicesOverX);',
		'    // dy2 = floor(s2/uSlicesOverY)/uSlicesOverY;',
		'    texpos1.x = dx1+(volpos.x/uSlicesOverX);',
		'    texpos1.y = dy1+(volpos.y/uSlicesOverY);',
		'    // texpos2.x = dx2+(volpos.x/uSlicesOverX);',
		'    // texpos2.y = dy2+(volpos.y/uSlicesOverY);',
		'    vec4 value;',
		'    float value1 = 0.0, value2 = 0.0; ',
		'    // bool value1Set = false, value2Set = false;',
		'    // int numberOfSlicemaps = int( ceil(uNumberOfSlices / (uSlicesOverX * uSlicesOverY)) );',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( tex1Index == <%=i%> )',
		'        {',
		'            //value1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            value = texture2D(uSliceMaps[<%=i%>],texpos1).rgba;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    return value;',
		'} ',
		'void main(void) {',
		'    vec2 texC = ((pos.xy / pos.w) + 1.0) / 2.0; ',
		'    vec4 backColor = texture2D(uBackCoord, texC); ',
		'    ',
		'    vec3 dir = backColor.rgb - frontColor.rgb; ',
		'    vec3 Step = dir / uSteps; ',
		'    vec4 vpos = frontColor;',
		'    vec4 accum = vec4(0, 0, 0, 0); ',
		'    vec4 sample = vec4(0.0, 0.0, 0.0, 0.0); ',
		'    vec4 colorValue = vec4(0, 0, 0, 0); ',
		'    vec4 gray_val = vec4(0.0, 0.0, 0.0, 0.0); ',
		' ',
		'    float biggest_gray_value = 0.0;',
		'    float opacityFactor = uOpacityVal; ',
		'    float lightFactor = uColorVal; ',
		'    for(float i=0.0; i<4095.0; i+=1.0) { ',
		'        if(i == uSteps) { ',
		'            break; ',
		'        }    ',
		'        gray_val = getVolumeValue(vpos.xyz); ',
		'        if(gray_val.x < uMinGrayVal || gray_val.x > uMaxGrayVal) { ',
		'            colorValue = vec4(0.0); ',
		'        } else { ',
		'            if(biggest_gray_value < gray_val.x) { ',
		'                biggest_gray_value = gray_val.x;',
		'            } ',
		'            if(uAbsorptionModeIndex == 0.0) { ',
		'                vec2 tf_pos; ',
		'                colorValue = gray_val;',
		'                sample.a = colorValue.a * opacityFactor; ',
		'                sample.rgb = colorValue.rgb * uColorVal; ',
		'                accum += sample; ',
		'                if(accum.a >= 1.0) ',
		'                    break; ',
		'            }',
		'            if(uAbsorptionModeIndex == 1.0) { ',
		'                colorValue = gray_val;',
		'                sample.a = colorValue.a * opacityFactor * (1.0 / uSteps); ',
		'                sample.rgb = (1.0 - accum.a) * colorValue.rgb * sample.a * lightFactor; ',
		'                accum += sample; ',
		'                if(accum.a >= 1.0) ',
		'                    break;',
		'            }',
		'        } ',
		'        //advance the current position ',
		'        vpos.xyz += Step; ',
		'        //break if the position is greater than <1, 1, 1> ',
		'        if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 ||',
		'           vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0)  { ',
		'            break; ',
		'        }',
		'    }',
		'    gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassBlending = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"darkness" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		"l" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'precision mediump int;',
		'precision mediump float;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uNumberOfSlices;',
		'uniform float uOpacityVal;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float darkness;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uSlicemapWidth;',
		'uniform float l;',
		'//Acts like a texture3D using Z slices and trilinear filtering.',
		'vec3 getVolumeValue(vec3 volpos) {',
		'  float s1Original, s2Original, s1, s2;',
		'  float dx1, dy1;',
		'  vec2 texpos1,texpos2;',
		'  float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'  s1Original = floor(volpos.z*uNumberOfSlices);',
		'  int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'  s1 = mod(s1Original, slicesPerSprite);',
		'  dx1 = fract(s1/uSlicesOverX);',
		'  dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'  texpos1.x = dx1+(volpos.x/uSlicesOverX);',
		'  texpos1.y = dy1+(volpos.y/uSlicesOverY);',
		'  vec3 value = vec3(0.0,0.0,0.0);',
		'  <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'      if( tex1Index == <%=i%> )',
		'      {',
		'          value = texture2D(uSliceMaps[<%=i%>],texpos1).xyz;',
		'      }',
		'      <% if( i < maxTexturesNumber-1 ) { %>',
		'          else',
		'      <% } %>',
		'  <% } %>',
		'  return value;',
		'}',
		'void main(void) {',
		'  const int uStepsI = 256;',
		'  const float uStepsF = float(uStepsI);',
		'  vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		'  vec4 backColor = texture2D(uBackCoord,texC);',
		'  vec3 dir = backColor.rgb - frontColor.rgb;',
		'  vec4 vpos = frontColor;',
		'  vec3 Step = dir/uStepsF;',
		'  vec4 accum = vec4(0, 0, 0, 0);',
		'  vec4 color = vec4(1.0);',
		'  float opacityFactor = uOpacityVal;',
		'  for(int i = 0; i < uStepsI; i++) {',
		'    vec3 gray_val = getVolumeValue(vpos.xyz);',
		'    if(gray_val.x > uMinGrayVal && gray_val.x < uMaxGrayVal) {',
		'      color.a = 0.04;',
		'      accum += color;',
		'      if(accum.a >= 1.0) break;',
		'    }',
		'    //advance the current position',
		'    vpos.xyz += Step;',
		'    if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0)',
		'        break;',
		'  }',
		'  gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassDVR = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"darkness" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		"l" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'precision mediump int;',
		'precision mediump float;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uNumberOfSlices;',
		'uniform float uOpacityVal;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float darkness;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uSlicemapWidth;',
		'uniform float l;',
		'//Acts like a texture3D using Z slices and trilinear filtering.',
		'vec3 getVolumeValue(vec3 volpos) {',
		'  float s1Original, s2Original, s1, s2;',
		'  float dx1, dy1;',
		'  vec2 texpos1,texpos2;',
		'  float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'  s1Original = floor(volpos.z*uNumberOfSlices);',
		'  int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'  s1 = mod(s1Original, slicesPerSprite);',
		'  dx1 = fract(s1/uSlicesOverX);',
		'  dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'  texpos1.x = dx1+(volpos.x/uSlicesOverX);',
		'  texpos1.y = dy1+(volpos.y/uSlicesOverY);',
		'  vec3 value = vec3(0.0,0.0,0.0);',
		'  <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'      if( tex1Index == <%=i%> )',
		'      {',
		'          value = texture2D(uSliceMaps[<%=i%>],texpos1).xyz;',
		'      }',
		'      <% if( i < maxTexturesNumber-1 ) { %>',
		'          else',
		'      <% } %>',
		'  <% } %>',
		'  return value;',
		'}',
		'void main(void) {',
		'  const int uStepsI = 256;',
		'  const float uStepsF = float(uStepsI);',
		'  vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		'  vec4 backColor = texture2D(uBackCoord,texC);',
		'  vec3 dir = backColor.rgb - frontColor.rgb;',
		'  vec4 vpos = frontColor;',
		'  vec3 Step = dir/uStepsF;',
		'  vec4 accum = vec4(0, 0, 0, 0);',
		'  vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);',
		'  vec4 colorValue = vec4(0, 0, 0, 0);',
		'  float opacityFactor = uOpacityVal;',
		'  for(int i = 0; i < uStepsI; i++) {',
		'    vec3 gray_val = getVolumeValue(vpos.xyz);',
		'    if(gray_val.z < 0.05 ||',
		'       gray_val.x < uMinGrayVal ||',
		'       gray_val.x > uMaxGrayVal)',
		'        colorValue = vec4(0.0);',
		'    else {',
		'      colorValue.x = (darkness * 2.0 - gray_val.x) * l * 0.4;',
		'      //colorValue.x = gray_val.x;',
		'      colorValue.w = 0.1;',
		'      sample.rgb = (1.0 - accum.a) * colorValue.xxx * sample.a;',
		'      sample.a = colorValue.a * opacityFactor * (1.0 / uStepsF);',
		'      accum += sample;',
		'      if(accum.a>=1.0)',
		'         break;',
		'    }',
		'    //advance the current position',
		'    vpos.xyz += Step;',
		'    if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0)',
		'        break;',
		'  }',
		'  gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassLidar = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uColormap" : { type: "t", value: null },
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'//see core.js -->',
		'//attributes: {',
		'//    vertColor: {type: \'c\', value: [] }',
		'//},',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'#ifdef GL_FRAGMENT_PRECISION_HIGH',
		' // highp is supported',
		' precision highp int;',
		' precision highp float;',
		'#else',
		' // high is not supported',
		' precision mediump int;',
		' precision mediump float;',
		'#endif',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uColormap;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'// returns total number of slices of all slicemaps',
		'uniform float uNumberOfSlices;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uOpacityVal;',
		'uniform float uColorVal;',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float uSlicemapWidth;',
		'float getVolumeValue(vec3 volpos)',
		'{',
		'    float value1 = 0.0;',
		'    vec2 texpos1;',
		'    vec3 value1_vec;',
		'    ',
		'    float eps =pow(2.0,-16.0);',
		'    if (volpos.x >= 1.0)',
		'        volpos.x = 1.0-eps;',
		'    if (volpos.y >= 1.0)',
		'        volpos.y = 1.0-eps;',
		'    if (volpos.z >= 1.0)',
		'        volpos.z = 1.0-eps;',
		'    ',
		'    float slicesPerSlicemap = uSlicesOverX * uSlicesOverY; ',
		'    float sliceNo = floor(volpos.z*(uNumberOfSlices));',
		'    ',
		'    int texIndexOfSlicemap = int(floor(sliceNo / slicesPerSlicemap));',
		'    float s1 = mod(sliceNo, slicesPerSlicemap);',
		'    float dx1 = fract(s1/uSlicesOverX);',
		'    float dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;      ',
		'       ',
		'    float sliceSizeX = uSlicemapWidth/uSlicesOverX;',
		'    float sliceSizeY = uSlicemapWidth/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(floor(volpos.x*sliceSizeX)+0.5)/uSlicemapWidth;',
		'    texpos1.y = dy1+(floor(volpos.y*sliceSizeY)+0.5)/uSlicemapWidth;',
		' ',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( texIndexOfSlicemap == <%=i%> )',
		'        {',
		'          value1_vec = texture2D(uSliceMaps[<%=i%>],texpos1).rgb;',
		'          //value1 = ((value1_vec.r + value1_vec.g + value1_vec.b)/3.0);',
		'          //value1 = ((value1_vec.r * 0.299)+(value1_vec.g * 0.587)+(value1_vec.b * 0.114));',
		'          value1 = value1_vec.r;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    ',
		'    return value1;',
		'}',
		'void main(void)',
		'{',
		' vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		' vec4 backColor = texture2D(uBackCoord,texC);',
		' vec3 dir = backColor.rgb - frontColor.rgb;',
		' vec4 vpos = frontColor;',
		' ',
		' ',
		' float dir_length = length(dir);',
		' float uStepsF = ceil((dir_length)*(uNumberOfSlices-1.0));',
		' vec3 Step = dir/(uStepsF);',
		' int uStepsI = int(uStepsF);',
		' ',
		' vec4 accum = vec4(0, 0, 0, 0);',
		' vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);',
		' vec4 colorValue = vec4(0, 0, 0, 0);',
		' float biggest_gray_value = 0.0;',
		' float opacityFactor = uOpacityVal;',
		' float lightFactor = uColorVal;',
		' ',
		' ',
		' ',
		' // Empty Skipping',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI) ',
		'         break;',
		' ',
		'     float gray_val = getVolumeValue(vpos.xyz);',
		'   ',
		'     if(gray_val <= uMinGrayVal || gray_val >= uMaxGrayVal) ',
		'         uStepsF -= 1.0;',
		'     ',
		'     vpos.xyz += Step;',
		'     ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'         break; ',
		' }',
		' vpos = frontColor;',
		' ',
		' ',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI) {',
		'         break;',
		'     }',
		'     float gray_val = getVolumeValue(vpos.xyz);',
		'     if(gray_val < uMinGrayVal || gray_val > uMaxGrayVal) {',
		'         colorValue = vec4(0.0);',
		'         accum=accum+colorValue;',
		'         if(accum.a>=1.0)',
		'            break;',
		'     } else {',
		'         // Stevens mode',
		'             vec2 tf_pos; ',
		'             tf_pos.x = (gray_val - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal); ',
		'             tf_pos.x = gray_val;',
		'             tf_pos.y = 0.5; ',
		'             colorValue = texture2D(uColormap,tf_pos);',
		'             //colorValue = texture2D(uTransferFunction,tf_pos);',
		'             //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0); ',
		'             sample.a = colorValue.a * opacityFactor * (1.0 / uStepsF); ',
		'             //sample.rgb = (1.0 - accum.a) * colorValue.rgb * sample.a * uColorVal; ',
		'             sample.rgb = colorValue.rgb; ',
		'             accum += sample; ',
		'             if(accum.a>=1.0) ',
		'                break; ',
		'     }',
		'     //advance the current position',
		'     vpos.xyz += Step;',
		'     ',
		'     //break if the position is greater than <1, 1, 1> ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'     { ',
		'         break; ',
		'     } ',
		'     ',
		' }',
		' gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassNearestNeighbour = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uSlicemapWidth" : { type: "f", value: -1 },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSteps" : { type: "f", value: -1 },
		"uZFactor" : { type: "f", value: -1 },
		"uAvailable_textures_number" : { type: "i", value: 0 },
		}
	]),
	vertexShader: [
		'varying vec3 worldSpaceCoords;',
		'varying vec4 projectedCoords;',
		' ',
		'void main()',
		'{',
		'    worldSpaceCoords = (modelMatrix * vec4(position + vec3(0.5, 0.5,0.5), 1.0 )).xyz;',
		'    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
		'    projectedCoords = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
		'}'].join("\n"),
	fragmentShader: [
		'#ifdef GL_FRAGMENT_PRECISION_HIGH ',
		' // highp is supported ',
		' precision highp int; ',
		' precision highp float; ',
		'#else ',
		' // high is not supported ',
		' precision mediump int; ',
		' precision mediump float; ',
		'#endif ',
		'// Passed from vertex',
		'varying vec3 worldSpaceCoords; ',
		'varying vec4 projectedCoords; ',
		'// Passed from core',
		'uniform sampler2D uBackCoord; ',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uSlicemapWidth;',
		'// Assuming a bounding box of 512x512x512',
		'// ceil( sqrt(3) * 512 ) = 887',
		'const int MAX_STEPS = 887;',
		'// Application specific parameters',
		'uniform float uNumberOfSlices; ',
		'uniform float uMinGrayVal; ',
		'uniform float uMaxGrayVal; ',
		'uniform float uOpacityVal; ',
		'uniform float uColorVal; ',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX; ',
		'uniform float uSlicesOverY; ',
		'uniform float uSteps;',
		'uniform float uZFactor;',
		'// uniform int uAvailable_textures_number;',
		'vec4 getVolumeValue(vec3 volpos)',
		'{',
		'    float s1Original, s2Original, s1, s2; ',
		'    float dx1, dy1; ',
		'    // float dx2, dy2; ',
		'    // float value; ',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    float sliceSizeX = uSlicemapWidth / uSlicesOverX;  // Number of pixels of ONE slice along x axis',
		'    float sliceSizeY = uSlicemapWidth / uSlicesOverY;  // Number of pixels of ONE slice along y axis',
		'    float delta = 1.0 / sliceSizeX;',
		'    ',
		'    float adapted_x, adapted_y, adapted_z;',
		'    adapted_x = (volpos.x * (1.0 - (2.0*delta))) + delta;',
		'    adapted_y = (volpos.y * (1.0 - (2.0*delta))) + delta;',
		'    adapted_z = 1.0 - ((volpos.z * (1.0 - (2.0*delta))) + delta);',
		'    s1Original = floor(adapted_z*uNumberOfSlices);',
		'    //s1Original = floor(volpos.z*uNumberOfSlices); ',
		'    // s2Original = min(s1Original + 1.0, uNumberOfSlices);',
		'    int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'    // int tex2Index = int(floor(s2Original / slicesPerSprite));',
		'    s1 = mod(s1Original, slicesPerSprite);',
		'    // s2 = mod(s2Original, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(floor(adapted_x*sliceSizeX)+0.5)/uSlicemapWidth;',
		'    texpos1.y = dy1+(floor(adapted_y*sliceSizeY)+0.5)/uSlicemapWidth;',
		' ',
		'    float value2 = 0.0;',
		'    vec4 value1;',
		'    // bool value1Set = false, value2Set = false;',
		'    // int numberOfSlicemaps = int( ceil(uNumberOfSlices / (uSlicesOverX * uSlicesOverY)) );',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( tex1Index == <%=i%> )',
		'        {',
		'            value1 = texture2D(uSliceMaps[<%=i%>],texpos1).rgba;',
		'        }',
		'    <% } %>',
		'    return value1;',
		'    // for (int x = 0; x < gl_MaxTextureImageUnits-2; x++)',
		'    // {',
		'    //     if(x == numberOfSlicemaps)',
		'    //     {',
		'    //         break;',
		'    //     }',
		'    //     if(x == tex1Index) { ',
		'    //         value1 = texture2D(uSliceMaps[x],texpos1).x; ',
		'    //         value1Set = true; ',
		'    //     } ',
		'    //     if(x == tex2Index) { ',
		'    //         value2 = texture2D(uSliceMaps[x],texpos2).x; ',
		'    //         value2Set = true; ',
		'    //     } ',
		'    //     if(value1Set && value2Set) { ',
		'    //         break; ',
		'    //     } ',
		'    // } ',
		'    // return mix(value1, value2, fract(volpos.z*uNumberOfSlices)); ',
		'} ',
		'void main(void) {',
		' ',
		'    //Transform the coordinates it from [-1;1] to [0;1]',
		'    vec2 texc = vec2(((projectedCoords.x / projectedCoords.w) + 1.0 ) / 2.0,',
		'                     ((projectedCoords.y / projectedCoords.w) + 1.0 ) / 2.0);',
		'    //The back position is the world space position stored in the texture.',
		'    vec3 backPos = texture2D(uBackCoord, texc).xyz;',
		'                ',
		'    //The front position is the world space position of the second render pass.',
		'    vec3 frontPos = worldSpaceCoords;',
		' ',
		'    //The direction from the front position to back position.',
		'    vec3 dir = backPos - frontPos;',
		'    //vec3 dir = frontPos - backPos;',
		' ',
		'    float rayLength = length(dir);',
		'    //Calculate how long to increment in each step.',
		'    float steps = ceil( sqrt(3.0) * (uSlicemapWidth / uSlicesOverX) ) * uZFactor;',
		'    float delta = 1.0 / steps;',
		'    ',
		'    //The increment in each direction for each step.',
		'    vec3 deltaDirection = normalize(dir) * delta;',
		'    float deltaDirectionLength = length(deltaDirection);',
		'    //Start the ray casting from the front position.',
		'    vec3 currentPosition = frontPos;',
		'    //The color accumulator.',
		'    vec4 accumulatedColor = vec4(0.0);',
		'    //The alpha value accumulated so far.',
		'    float accumulatedAlpha = 0.0;',
		'    ',
		'    //How long has the ray travelled so far.',
		'    float accumulatedLength = 0.0;',
		'    ',
		'    //If we have twice as many samples, we only need ~1/2 the alpha per sample.',
		'    //Scaling by 256/10 just happens to give a good value for the alphaCorrection slider.',
		'    float alphaScaleFactor = 25.6 * delta;',
		'    ',
		'    vec4 colorSample;',
		'    float alphaSample;',
		'    float alphaCorrection = 1.0;',
		'    ',
		'    //Perform the ray marching iterations',
		'    for(int i = 0; i < MAX_STEPS; i++) {',
		'        //Get the voxel intensity value from the 3D texture.',
		'        //colorSample = sampleAs3DTexture( currentPosition );',
		'        ',
		'        colorSample = getVolumeValue( currentPosition );',
		'        ',
		'        //Allow the alpha correction customization.',
		'        alphaSample = colorSample.a * alphaCorrection;',
		'        ',
		'        //Applying this effect to both the color and alpha accumulation results in more realistic transparency.',
		'        alphaSample *= (1.0 - accumulatedAlpha);',
		'        ',
		'        //Scaling alpha by the number of steps makes the final color invariant to the step size.',
		'        alphaSample *= alphaScaleFactor;',
		'        ',
		'        //Perform the composition.',
		'        accumulatedColor += colorSample * alphaSample;',
		'        ',
		'        //Store the alpha accumulated so far.',
		'        accumulatedAlpha += alphaSample;',
		'        ',
		'        //Advance the ray.',
		'        currentPosition += deltaDirection;',
		'					',
		'        accumulatedLength += deltaDirectionLength;',
		'        ',
		'        //If the length traversed is more than the ray length, or if the alpha accumulated reaches 1.0 then exit.',
		'        if(accumulatedLength >= rayLength || accumulatedAlpha >= 1.0 )',
		'            break;',
		'    }',
		'    gl_FragColor = accumulatedColor; ',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassNearestNeighbourRGB = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uSlicemapWidth" : { type: "f", value: -1 },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSteps" : { type: "f", value: -1 },
		"uRatio" : { type: "f", value: -1 },
		"uAvailable_textures_number" : { type: "i", value: 0 },
		}
	]),
	vertexShader: [
		'varying vec3 worldSpaceCoords;',
		'varying vec4 projectedCoords;',
		' ',
		'void main()',
		'{',
		'    worldSpaceCoords = (modelMatrix * vec4(position + vec3(0.5, 0.5,0.5), 1.0 )).xyz;',
		'    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
		'    projectedCoords = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
		'}'].join("\n"),
	fragmentShader: [
		'#ifdef GL_FRAGMENT_PRECISION_HIGH ',
		' // highp is supported ',
		' precision highp int; ',
		' precision highp float; ',
		'#else ',
		' // high is not supported ',
		' precision mediump int; ',
		' precision mediump float; ',
		'#endif ',
		'// Passed from vertex',
		'varying vec3 worldSpaceCoords; ',
		'varying vec4 projectedCoords; ',
		'// Passed from core',
		'uniform sampler2D uBackCoord; ',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uSlicemapWidth;',
		'// Assuming a bounding box of 512x512x512',
		'// ceil( sqrt(3) * 512 ) = 887',
		'const int MAX_STEPS = 887;',
		'// Application specific parameters',
		'uniform float uNumberOfSlices; ',
		'uniform float uMinGrayVal; ',
		'uniform float uMaxGrayVal;',
		'uniform float uOpacityVal; ',
		'uniform float uColorVal; ',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX; ',
		'uniform float uSlicesOverY; ',
		'uniform float uSteps;',
		'uniform float uRatio;',
		'// uniform int uAvailable_textures_number;',
		'vec4 getVolumeValue(vec3 volpos)',
		'{',
		'    //if (volpos.z < 0.5)',
		'    //    return vec4(0.0);',
		'    float s1Original, s2Original, s1, s2; ',
		'    float dx1, dy1; ',
		'    // float dx2, dy2; ',
		'    // float value; ',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    float sliceSizeX = uSlicemapWidth / uSlicesOverX;  // Number of pixels of ONE slice along x axis',
		'    float sliceSizeY = uSlicemapWidth / uSlicesOverY;  // Number of pixels of ONE slice along y axis',
		'    float delta = 1.0 / sliceSizeX;',
		'    ',
		'    float adapted_x, adapted_y, adapted_z;',
		'    adapted_x = (volpos.x * (1.0 - (2.0*delta))) + delta;',
		'    adapted_y = (volpos.y * (1.0 - (2.0*delta))) + delta;',
		'    adapted_z = 1.0 - (( (volpos.z* (1.0/uRatio) ) * (1.0 - (2.0*delta))) + delta);',
		'    s1Original = floor(adapted_z*uNumberOfSlices);',
		'    //s1Original = floor(volpos.z*uNumberOfSlices); ',
		'    //s2Original = min(s1Original + 1.0, uNumberOfSlices);',
		'    int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'    //int tex2Index = int(floor(s2Original / slicesPerSprite));',
		'    s1 = mod(s1Original, slicesPerSprite);',
		'    //s2 = mod(s2Original, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(floor(adapted_x*sliceSizeX)+0.5)/uSlicemapWidth;',
		'    texpos1.y = dy1+(floor(adapted_y*sliceSizeY)+0.5)/uSlicemapWidth;',
		' ',
		'    float value2 = 0.0;',
		'    vec4 value1;',
		'    // bool value1Set = false, value2Set = false;',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( tex1Index == <%=i%> )',
		'        {',
		'            value1 = texture2D(uSliceMaps[<%=i%>],texpos1).rgba;',
		'        }',
		'    <% } %>',
		'    return value1;',
		'    // for (int x = 0; x < gl_MaxTextureImageUnits-2; x++)',
		'    // {',
		'    //     if(x == numberOfSlicemaps)',
		'    //     {',
		'    //         break;',
		'    //     }',
		'    //     if(x == tex1Index) { ',
		'    //         value1 = texture2D(uSliceMaps[x],texpos1).x; ',
		'    //         value1Set = true; ',
		'    //     } ',
		'    //     if(x == tex2Index) { ',
		'    //         value2 = texture2D(uSliceMaps[x],texpos2).x; ',
		'    //         value2Set = true; ',
		'    //     } ',
		'    //     if(value1Set && value2Set) { ',
		'    //         break; ',
		'    //     } ',
		'    // } ',
		'    // return mix(value1, value2, fract(volpos.z*uNumberOfSlices));',
		'}',
		'void main(void) {',
		' ',
		'    //Transform the coordinates it from [-1;1] to [0;1]',
		'    vec2 texc = vec2(((projectedCoords.x / projectedCoords.w) + 1.0 ) / 2.0,',
		'                     ((projectedCoords.y / projectedCoords.w) + 1.0 ) / 2.0);',
		'    //The back position is the world space position stored in the texture.',
		'    vec3 backPos = texture2D(uBackCoord, texc).xyz;',
		'                ',
		'    //The front position is the world space position of the second render pass.',
		'    vec3 frontPos = worldSpaceCoords;',
		' ',
		'    //The direction from the front position to back position.',
		'    vec3 dir = backPos - frontPos;',
		'    float rayLength = length(dir);',
		'    //Calculate how long to increment in each step.',
		'    float steps = ceil( sqrt(3.0) * (uSlicemapWidth / uSlicesOverX) ) * uRatio;',
		'    //float steps = 256.0;',
		'    float delta = 1.0 / steps;',
		'    ',
		'    //The increment in each direction for each step.',
		'    vec3 deltaDirection = normalize(dir) * delta;',
		'    ',
		'    vec3 Step = dir / steps;',
		'    ',
		'    float deltaDirectionLength = length(deltaDirection);',
		'    //vec4 vpos = frontColor;  // currentPosition',
		'    //vec3 Step = dir/uStepsF; // steps',
		'    //Start the ray casting from the front position.',
		'    vec3 currentPosition = frontPos;',
		'    //The color accumulator.',
		'    vec4 accumulatedColor = vec4(0.0);',
		'    //The alpha value accumulated so far.',
		'    float accumulatedAlpha = 0.0;',
		'    ',
		'    //How long has the ray travelled so far.',
		'    float accumulatedLength = 0.0;',
		'    ',
		'    //If we have twice as many samples, we only need ~1/2 the alpha per sample.',
		'    //Scaling by 256/10 just happens to give a good value for the alphaCorrection slider.',
		'    float alphaScaleFactor = 28.8 * delta;',
		'    ',
		'    vec4 colorSample = vec4(0.0);',
		'    vec4 sample = vec4(0.0); ',
		'    vec4 grayValue;',
		'    float alphaSample;',
		'    float alphaCorrection = 1.0;',
		'    ',
		'    //Perform the ray marching iterations',
		'    for(int i = 0; i < MAX_STEPS; i++) {       ',
		'        if(currentPosition.x > 1.0 || currentPosition.y > 1.0 || currentPosition.z > 1.0 || currentPosition.x < 0.0 || currentPosition.y < 0.0 || currentPosition.z < 0.0)      ',
		'            break;',
		'        if(accumulatedColor.a>=1.0) ',
		'            break;',
		'        grayValue = getVolumeValue(currentPosition); ',
		'        if(grayValue.z < 0.05 || ',
		'           grayValue.x < 0.0 ||',
		'           grayValue.x > 1.0)  ',
		'            accumulatedColor = vec4(0.0);     ',
		'        else { ',
		'            //colorSample.x = (1.0 * 2.0 - grayValue.x) * 5.0 * 0.4;',
		'            colorSample.xyz = grayValue.xyz;',
		'            //colorSample.w = alphaScaleFactor;',
		'            colorSample.w = 0.1;',
		'              ',
		'            //sample.a = colorSample.a * 40.0 * (1.0 / steps);',
		'            sample.a = colorSample.a;',
		'            sample.rgb = (1.0 - accumulatedColor.a) * colorSample.xyz * sample.a; ',
		'             ',
		'            accumulatedColor += sample; ',
		'        }    ',
		'   ',
		'        //Advance the ray.',
		'        //currentPosition.xyz += deltaDirection;',
		'        currentPosition.xyz += Step;',
		'   ',
		'         ',
		'    } ',
		'    gl_FragColor = accumulatedColor;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassSoebel = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"darkness" : { type: "f", value: -1 },
		"uLightPos" : { type: "v3", value: new THREE.Vector3( 0, 0, 0 ) },
		"uSetViewMode" : { type: "i", value: 0 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		"l" : { type: "f", value: -1 },
		"s" : { type: "f", value: -1 },
		"hMin" : { type: "f", value: -1 },
		"hMax" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'// This is an experimental shader to implement',
		'// blinn phong shading model.',
		'// In this example, I use the USCT breast model',
		'// with a total of 144 slices as the dataset.',
		'// Hence the gradient operator is divided by 144 for',
		'// a single unit. Uncomment line 271 to see the normals',
		'// calculated by the gradient operator function.',
		'precision mediump int;',
		'precision mediump float;',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uNumberOfSlices;',
		'uniform float uOpacityVal;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float darkness;',
		'uniform vec3 uLightPos;',
		'uniform int uSetViewMode;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uSlicemapWidth;',
		'uniform float l;',
		'uniform float s;',
		'uniform float hMin;',
		'uniform float hMax;',
		'//Acts like a texture3D using Z slices and trilinear filtering.',
		'vec3 getVolumeValue(vec3 volpos)',
		'{',
		'    if ( (volpos.x < 1.0/255.0) || (volpos.x > (1.0 - 1.0/255.0)) ) {',
		'        return vec3(0.0);',
		'    }',
		'    if ( (volpos.y < 1.0/255.0) || (volpos.y > (1.0 - 1.0/255.0)) ) {',
		'        return vec3(0.0);',
		'    }',
		'    if ( (volpos.z < 1.0/255.0) || (volpos.z > (1.0 - 1.0/255.0)) ) {',
		'        return vec3(0.0);',
		'    }',
		'    float s1Original, s2Original, s1, s2;',
		'    float dx1, dy1;',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    s1Original = floor(volpos.z*uNumberOfSlices);',
		'    int tex1Index = int(floor(s1Original / slicesPerSprite));',
		'    s1 = mod(s1Original, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(volpos.x/uSlicesOverX);',
		'    texpos1.y = dy1+(volpos.y/uSlicesOverY);',
		'    vec3 value = vec3(0.0,0.0,0.0);',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( tex1Index == <%=i%> )',
		'        {',
		'            value = texture2D(uSliceMaps[<%=i%>],texpos1).xyz;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    return value;',
		'}',
		'// Compute the Normal around the current voxel',
		'vec3 getNormal(vec3 at)',
		'{',
		'    float xw = uSlicemapWidth / uSlicesOverX;',
		'    float yw = uSlicemapWidth / uSlicesOverY;',
		'    float zw = uNumberOfSlices;',
		'    float fSliceLower, fSliceUpper, s1, s2;',
		'    float dx1, dy1, dx2, dy2;',
		'    int iTexLowerIndex, iTexUpperIndex;',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    fSliceLower = floor(at.z*uNumberOfSlices); // z value is between 0 and 1. Multiplying the total number of slices',
		'                                               // gives the position in between. By flooring the value, you get the lower',
		'                                               // slice position.',
		'    fSliceUpper = min(fSliceLower + 1.0, uNumberOfSlices); // return the mininimum between the two values',
		'                                                           // act as a upper clamp.',
		'    // At this point, we get our lower slice and upper slice',
		'    // Now we need to get which texture image contains our slice.',
		'    iTexLowerIndex = int(floor(fSliceLower / slicesPerSprite));',
		'    iTexUpperIndex = int(floor(fSliceUpper / slicesPerSprite));',
		'    // mod returns the value of x modulo y. This is computed as x - y * floor(x/y).',
		'    s1 = mod(fSliceLower, slicesPerSprite); // returns the index of slice in slicemap',
		'    s2 = mod(fSliceUpper, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    dx2 = fract(s2/uSlicesOverX);',
		'    dy2 = floor(s2/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    float weight = at.z - floor(at.z);',
		'    float w1 = at.z - floor(at.z);',
		'    float w0 = (at.z - (1.0/zw)) - floor(at.z);',
		'    float w2 = (at.z + (1.0/zw)) - floor(at.z);',
		'    float fx, fy, fz;',
		'    float L0, L1, L2, L3, L4, L5, L6, L7, L8;',
		'    float H0, H1, H2, H3, H4, H5, H6, H7, H8;',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( iTexLowerIndex == <%=i%> )',
		'        {',
		'            texpos1.x = dx1+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/yw)/uSlicesOverY);',
		'            L0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/yw)/uSlicesOverY);',
		'            L1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/yw)/uSlicesOverY);',
		'            L2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/yw)/uSlicesOverY);',
		'            L3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/yw)/uSlicesOverY);',
		'            L4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/yw)/uSlicesOverY);',
		'            L5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/yw)/uSlicesOverY);',
		'            L6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/yw)/uSlicesOverY);',
		'            L7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx1+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/yw)/uSlicesOverY);',
		'            L8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        }',
		'        if( iTexUpperIndex == <%=i%> ) {',
		'            texpos1.x = dx2+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/yw)/uSlicesOverY);',
		'            H0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/yw)/uSlicesOverY);',
		'            H1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/yw)/uSlicesOverY);',
		'            H2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/yw)/uSlicesOverY);',
		'            H3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/yw)/uSlicesOverY);',
		'            H4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/yw)/uSlicesOverY);',
		'            H5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x - 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/yw)/uSlicesOverY);',
		'            H6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 0.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/yw)/uSlicesOverY);',
		'            H7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            texpos1.x = dx2+((at.x + 1.0/xw)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/yw)/uSlicesOverY);',
		'            H8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        }',
		'    <% } %>',
		'    // we need to get interpolation of 2 x points',
		'    // x direction',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // -3 -6 -3   0  0  0   3  6  3',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // y direction',
		'    //  1  3  1   3  6  3   1  3  1',
		'    //  0  0  0   0  0  0   0  0  0',
		'    // -1 -3 -1  -3 -6 -3  -1 -3 -1',
		'    // z direction',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    // -3  0  3   -6  0  6   -3  0  3',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    fx =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * -3.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w0 * (H3 - L3)) + L3) * -3.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * -6.0;',
		'    fx += ((w2 * (H3 - L3)) + L3) * -3.0;',
		'    fx += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * -3.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w0 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w0 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 3.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w0 * (H5 - L5)) + L5) * 3.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 6.0;',
		'    fx += ((w2 * (H5 - L5)) + L5) * 3.0;',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 3.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 3.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w0 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * -3.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w0 * (H1 - L1)) + L1) * 3.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 6.0;',
		'    fy += ((w2 * (H1 - L1)) + L1) * 3.0;',
		'    fy += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w0 * (H7 - L7)) + L7) * -3.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * -6.0;',
		'    fy += ((w2 * (H7 - L7)) + L7) * -3.0;',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 3.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w0 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * -3.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * -1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 0.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    fz += ((w0 * (H3 - L3)) + L3) * -3.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 3.0;',
		'    fz += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 0.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    fz += ((w0 * (H1 - L1)) + L1) * -3.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 3.0;',
		'    fz += ((w0 * (H4 - L4)) + L4) * -6.0;',
		'    fz += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 6.0;',
		'    fz += ((w0 * (H7 - L7)) + L7) * -3.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 3.0;',
		'    fz += ((w0 * (H2 - L2)) + L2) * -1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 0.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    fz += ((w0 * (H5 - L5)) + L5) * -3.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 3.0;',
		'    fz += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 0.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    vec3 n = vec3( fx/27.0 , fy/27.0 , fz/27.0 );',
		'    return n;',
		'}',
		'// returns intensity of reflected ambient lighting',
		'const vec3 lightColor = vec3(1.0, 0.88, 0.74);',
		'const vec3 u_intensity = vec3(0.1, 0.1, 0.1);',
		'vec3 ambientLighting()',
		'{',
		'    const vec3 u_matAmbientReflectance = lightColor;',
		'    const vec3 u_lightAmbientIntensity = u_intensity;',
		'    return u_matAmbientReflectance * u_lightAmbientIntensity;',
		'}',
		'// returns intensity of diffuse reflection',
		'vec3 diffuseLighting(in vec3 N, in vec3 L)',
		'{',
		'    const vec3 u_matDiffuseReflectance = lightColor;',
		'    const vec3 u_lightDiffuseIntensity = vec3(0.6, 0.6, 0.6);',
		'    // calculation as for Lambertian reflection',
		'    float diffuseTerm = dot(N, L);',
		'    if (diffuseTerm > 1.0) {',
		'        diffuseTerm = 1.0;',
		'    } else if (diffuseTerm < 0.0) {',
		'        diffuseTerm = 0.0;',
		'    }',
		'    return u_matDiffuseReflectance * u_lightDiffuseIntensity * diffuseTerm;',
		'}',
		'// returns intensity of specular reflection',
		'vec3 specularLighting(in vec3 N, in vec3 L, in vec3 V)',
		'{',
		'  float specularTerm = 0.0;',
		'    // const vec3 u_lightSpecularIntensity = vec3(0, 1, 0);',
		'    const vec3 u_lightSpecularIntensity = u_intensity;',
		'    const vec3 u_matSpecularReflectance = lightColor;',
		'    const float u_matShininess = 5.0;',
		'   // calculate specular reflection only if',
		'   // the surface is oriented to the light source',
		'   if(dot(N, L) > 0.0)',
		'   {',
		'      vec3 e = normalize(-V);',
		'      vec3 r = normalize(-reflect(L, N));',
		'      specularTerm = pow(max(dot(r, e), 0.0), u_matShininess);',
		'   }',
		'   return u_matSpecularReflectance * u_lightSpecularIntensity * specularTerm;',
		'}',
		'void main(void)',
		'{',
		'    float xw = uSlicemapWidth / uSlicesOverX;',
		'    float yw = uSlicemapWidth / uSlicesOverY;',
		'    float zw = uNumberOfSlices;',
		'    const int uStepsI = 256;',
		'    const float uStepsF = float(uStepsI);',
		'    vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		'    vec4 backColor = texture2D(uBackCoord,texC);',
		'    vec3 dir = backColor.rgb - frontColor.rgb;',
		'    vec4 vpos = frontColor;',
		'    vec3 Step = dir/uStepsF;',
		'    vec4 accum = vec4(0, 0, 0, 0);',
		'    vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);',
		'    vec4 colorValue = vec4(0, 0, 0, 0);',
		'    float opacityFactor = uOpacityVal;',
		'    vec3 lightPos[3];',
		'    lightPos[0] = vec3(1, 1, 1);',
		'    lightPos[1] = vec3(-1, -1, -1);',
		'    lightPos[2] = vec3(1, 1, -1);',
		'    // float xsqu;',
		'    // float ysqu;',
		'    // float distanceFromCenter;',
		'    for(int i = 0; i < uStepsI; i++) {',
		'      // xsqu = (0.5 - vpos.x) * (0.5 - vpos.x);',
		'      // ysqu = (0.5 - vpos.y) * (0.5 - vpos.y);',
		'      // distanceFromCenter = sqrt(xsqu + ysqu);',
		'      //',
		'      // if (distanceFromCenter < 0.4534 && vpos.z > 0.1 && vpos.z < 0.9) {',
		'        vec3 gray_val = getVolumeValue(vpos.xyz);',
		'        /************************************/',
		'        /*         Mean filtering           */',
		'        /************************************/',
		'        /*',
		'        if (gray_val.x > uMinGrayVal && gray_val.x < uMaxGrayVal) {',
		'          float sum_gray_val = 0.0;',
		'          int mask_size = 3;',
		'          vec3 offset;',
		'          vec3 curDotPos;',
		'          for(int m_i = 0; m_i < 3; ++m_i) { // 3 = mask_size',
		'            for(int j = 0; j < 3; ++j) {',
		'              for(int k = 0; k < 3; ++k) {',
		'                offset = vec3((float(m_i) - 1.0) / xw, // 1.0 = (int)mask_size / 2',
		'                              (float(j) - 1.0) / yw,',
		'                              (float(k) - 1.0) / zw);',
		'                curDotPos = vpos.xyz + offset;',
		'                sum_gray_val += getVolumeValue(curDotPos).x;',
		'              }',
		'            }',
		'          }',
		'          gray_val.x = sum_gray_val / 27.0; // 27.0 = pow(mask_size, 3)',
		'        } // end of Mean filtering',
		'        */',
		'        ',
		'        if(gray_val.z < 0.00 ||',
		'           gray_val.x < uMinGrayVal ||',
		'           gray_val.x > uMaxGrayVal) {',
		'            colorValue = vec4(0.0);',
		'        } else {',
		'            /* surface rendering',
		'            vec3 V = normalize(cameraPosition - vpos.xyz);',
		'            vec3 N = normalize(getNormal(vpos.xyz));',
		'            for(int light_i = 0; light_i < 3; ++light_i) {',
		'              vec3 L = normalize(lightPos[light_i] - vpos.xyz);',
		'              vec3 Iamb = ambientLighting();',
		'              vec3 Idif = diffuseLighting(N, L);',
		'              vec3 Ispe = specularLighting(N, L, V);',
		'              sample.rgb += (Iamb + Idif + Ispe);',
		'            }',
		'            sample.a = 1.0;',
		'            */',
		'            ',
		'            if ( uSetViewMode == 1 ) {',
		'                vec3 V = normalize(cameraPosition - vpos.xyz);',
		'                vec3 N = normalize(getNormal(vpos.xyz));',
		'                for(int light_i = 0; light_i < 3; ++light_i) {',
		'                    vec3 L = normalize(lightPos[light_i] - vpos.xyz);',
		'                    vec3 Iamb = ambientLighting();',
		'                    vec3 Idif = diffuseLighting(N, L);',
		'                    vec3 Ispe = specularLighting(N, L, V);',
		'                    sample.rgb += (Iamb + Idif + Ispe);',
		'                }',
		'                sample.a = 1.0;',
		'            } else {',
		'                //float test = (darkness * 2.0 - gray_val.x) * l * 0.4;',
		'                colorValue = texture2D(uTransferFunction, vec2(gray_val.x, 0.5));',
		'                //colorValue.x = gray_val.x;',
		'                colorValue.w = 0.1;',
		'                sample.rgb = (1.0 - accum.a) * colorValue.xyz * sample.a;',
		'                sample.a = colorValue.a * opacityFactor * (1.0 / uStepsF);',
		'            }',
		'            ',
		'            accum += sample;',
		'            ',
		'            if(accum.a>=1.0)',
		'               break;',
		'        }',
		'        //advance the current position',
		'        vpos.xyz += Step;',
		'        if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > (1.0 - pow(2.0,-16.0))|| vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0)',
		'            break;',
		'    }',
		'    gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassSoebelNTJ = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uColormap" : { type: "t", value: null },
		"uBackCoord" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"darkness" : { type: "f", value: -1 },
		"uLightPos" : { type: "v3", value: new THREE.Vector3( 0, 0, 0 ) },
		"uSetViewMode" : { type: "i", value: 0 },
		"uSteps" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"minSos" : { type: "f", value: -1 },
		"maxSos" : { type: "f", value: -1 },
		"l" : { type: "f", value: -1 },
		"s" : { type: "f", value: -1 },
		"hMin" : { type: "f", value: -1 },
		"hMax" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int; ',
		'precision mediump float; ',
		'attribute vec4 vertColor; ',
		'varying vec4 frontColor; ',
		'varying vec4 pos; ',
		'void main(void) ',
		'{ ',
		'    frontColor = vertColor; ',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0); ',
		'    gl_Position = pos; ',
		'} '].join("\n"),
	fragmentShader: [
		'// This is an experimental shader to implement',
		'// blinn phong shading model.',
		'// In this example, I use the USCT breast model ',
		'// with a total of 144 slices as the dataset.',
		'// Hence the gradient operator is divided by 144 for ',
		'// a single unit. Uncomment line 271 to see the normals',
		'// calculated by the gradient operator function.',
		'//precision mediump int; ',
		'//precision mediump float;',
		'varying vec4 frontColor; ',
		'varying vec4 pos; ',
		'uniform sampler2D uColormap; ',
		'uniform sampler2D uBackCoord; ',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'uniform float uNumberOfSlices; ',
		'uniform float uOpacityVal; ',
		'uniform float uSlicesOverX; ',
		'uniform float uSlicesOverY; ',
		'uniform float darkness;',
		'uniform vec3 uLightPos;',
		'uniform int uSetViewMode;',
		'uniform float uSteps;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float minSos;',
		'uniform float maxSos;',
		'uniform float l; ',
		'uniform float s; ',
		'uniform float hMin; ',
		'uniform float hMax; ',
		'// returns total number of slices of all slicemaps',
		'uniform float uSlicemapWidth;',
		'float getVolumeValue(vec3 volpos)',
		'{',
		'    float value1 = 0.0;',
		'    vec2 texpos1;',
		'    vec3 value1_vec;',
		'    ',
		'    float eps =pow(2.0,-16.0);',
		'    if (volpos.x >= 1.0)',
		'        volpos.x = 1.0-eps;',
		'    if (volpos.y >= 1.0)',
		'        volpos.y = 1.0-eps;',
		'    if (volpos.z >= 1.0)',
		'        volpos.z = 1.0-eps;',
		'    ',
		'    float slicesPerSlicemap = uSlicesOverX * uSlicesOverY; ',
		'    float sliceNo = floor(volpos.z*(uNumberOfSlices));',
		'    ',
		'    int texIndexOfSlicemap = int(floor(sliceNo / slicesPerSlicemap));',
		'    float s1 = mod(sliceNo, slicesPerSlicemap);',
		'    float dx1 = fract(s1/uSlicesOverX);',
		'    float dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;      ',
		'       ',
		'    float sliceSizeX = uSlicemapWidth/uSlicesOverX;',
		'    float sliceSizeY = uSlicemapWidth/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(floor(volpos.x*sliceSizeX)+0.5)/uSlicemapWidth;',
		'    texpos1.y = dy1+(floor(volpos.y*sliceSizeY)+0.5)/uSlicemapWidth;',
		' ',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( texIndexOfSlicemap == <%=i%> )',
		'        {',
		'          value1_vec = texture2D(uSliceMaps[<%=i%>],texpos1).rgb;',
		'          //value1 = ((value1_vec.r + value1_vec.g + value1_vec.b)/3.0);',
		'          //value1 = ((value1_vec.r * 0.299)+(value1_vec.g * 0.587)+(value1_vec.b * 0.114));',
		'          value1 = value1_vec.r;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    ',
		'    return value1;',
		'}',
		'/*',
		'//Acts like a texture3D using Z slices and trilinear filtering. ',
		'vec3 getVolumeValue(vec3 volpos)',
		'{',
		'    float s1Original, s2Original, s1, s2; ',
		'    float dx1, dy1; ',
		'    vec2 texpos1,texpos2; ',
		'    float eps =pow(2.0,-16.0);',
		'    if (volpos.x >= 1.0)',
		'        volpos.x = 1.0-eps;',
		'    if (volpos.y >= 1.0)',
		'        volpos.y = 1.0-eps;',
		'    if (volpos.z >= 1.0)',
		'        volpos.z = 1.0-eps;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY; ',
		'    s1Original = floor(volpos.z*uNumberOfSlices);     ',
		'    int tex1Index = int(floor(s1Original / slicesPerSprite));    ',
		'    s1 = mod(s1Original, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    texpos1.x = dx1+(volpos.x/uSlicesOverX);',
		'    texpos1.y = dy1+(volpos.y/uSlicesOverY);',
		'    vec3 value = vec3(0.0,0.0,0.0); ',
		'    ',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( tex1Index == <%=i%> )',
		'        {',
		'            value = texture2D(uSliceMaps[<%=i%>],texpos1).xyz;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    return value;',
		'}',
		'*/',
		'// Compute the Normal around the current voxel',
		'vec3 getVolumeValue_Soebel(vec3 at)',
		'{',
		'    float fSliceLower, fSliceUpper, s1, s2;',
		'    float dx1, dy1, dx2, dy2;',
		'    int iTexLowerIndex, iTexUpperIndex;',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    fSliceLower = floor(at.z*uNumberOfSlices); // z value is between 0 and 1. Multiplying the total number of slices',
		'                                               // gives the position in between. By flooring the value, you get the lower',
		'                                               // slice position.',
		'    fSliceUpper = min(fSliceLower + 1.0, uNumberOfSlices); // return the mininimum between the two values',
		'                                                           // act as a upper clamp.',
		'    // At this point, we get our lower slice and upper slice',
		'    // Now we need to get which texture image contains our slice.',
		'    iTexLowerIndex = int(floor(fSliceLower / slicesPerSprite));',
		'    iTexUpperIndex = int(floor(fSliceUpper / slicesPerSprite));',
		'    // mod returns the value of x modulo y. This is computed as x - y * floor(x/y).',
		'    s1 = mod(fSliceLower, slicesPerSprite); // returns the index of slice in slicemap',
		'    s2 = mod(fSliceUpper, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    dx2 = fract(s2/uSlicesOverX);',
		'    dy2 = floor(s2/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    float weight = at.z - floor(at.z);',
		'    float w1 = at.z - floor(at.z);',
		'    float w0 = (at.z - (1.0/uNumberOfSlices)) - floor(at.z);',
		'    float w2 = (at.z + (1.0/uNumberOfSlices)) - floor(at.z);',
		'    ',
		'    ',
		'    float fx, fy, fz;',
		'    ',
		'    float L0, L1, L2, L3, L4, L5, L6, L7, L8;',
		'    float H0, H1, H2, H3, H4, H5, H6, H7, H8;',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( iTexLowerIndex == <%=i%> )',
		'        {',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'        }',
		'        if( iTexUpperIndex == <%=i%> ) {',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        }',
		'    <% } %>',
		'    float value = (L4*(1.0-weight)) + (H4*(weight));',
		'    vec3 n = vec3(value);',
		'    return n;',
		'}',
		'// Compute the Normal around the current voxel',
		'vec3 getNormalSmooth(vec3 at)',
		'{',
		'    float fSliceLower, fSliceUpper, s1, s2;',
		'    float dx1, dy1, dx2, dy2;',
		'    int iTexLowerIndex, iTexUpperIndex;',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    fSliceLower = floor(at.z*uNumberOfSlices); // z value is between 0 and 1. Multiplying the total number of slices',
		'                                               // gives the position in between. By flooring the value, you get the lower',
		'                                               // slice position.',
		'    fSliceUpper = min(fSliceLower + 1.0, uNumberOfSlices); // return the mininimum between the two values',
		'                                                           // act as a upper clamp.',
		'    // At this point, we get our lower slice and upper slice',
		'    // Now we need to get which texture image contains our slice.',
		'    iTexLowerIndex = int(floor(fSliceLower / slicesPerSprite));',
		'    iTexUpperIndex = int(floor(fSliceUpper / slicesPerSprite));',
		'    // mod returns the value of x modulo y. This is computed as x - y * floor(x/y).',
		'    s1 = mod(fSliceLower, slicesPerSprite); // returns the index of slice in slicemap',
		'    s2 = mod(fSliceUpper, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    dx2 = fract(s2/uSlicesOverX);',
		'    dy2 = floor(s2/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    float weight = at.z - floor(at.z);',
		'    float w1 = at.z - floor(at.z);',
		'    float w0 = (at.z - (1.0/uNumberOfSlices)) - floor(at.z);',
		'    float w2 = (at.z + (1.0/uNumberOfSlices)) - floor(at.z);',
		'    ',
		'    ',
		'    float fx, fy, fz;',
		'    ',
		'    float L0, L1, L2, L3, L4, L5, L6, L7, L8;',
		'    float H0, H1, H2, H3, H4, H5, H6, H7, H8;',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( iTexLowerIndex == <%=i%> )',
		'        {',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'        }',
		'        if( iTexUpperIndex == <%=i%> ) {',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        }',
		'    <% } %>',
		'    // we need to get interpolation of 2 x points',
		'    // x direction',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // -3 -6 -3   0  0  0   3  6  3',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // y direction',
		'    //  1  3  1   3  6  3   1  3  1',
		'    //  0  0  0   0  0  0   0  0  0',
		'    // -1 -3 -1  -3 -6 -3  -1 -3 -1',
		'    // z direction',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    // -3  0  3   -6  0  6   -3  0  3',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    /*',
		'    fx =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * -2.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * -1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * -4.0; //-4.0',
		'    fx += ((w2 * (H3 - L3)) + L3) * -2.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 0.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 0.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 2.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 4.0; //4.0',
		'    fx += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 2.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 2.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 0.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 2.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 4.0; // 4.0',
		'    fy += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * -4.0; // -4.0',
		'    fy += ((w2 * (H7 - L7)) + L7) * -2.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 0.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * -2.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * -1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 0.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 2.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 0.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * -2.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * -4.0; //-4.0',
		'    fz += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 4.0; // 4.0',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 2.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * -1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 0.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * -2.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 0.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    */',
		'    /*',
		'    fx =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * -2.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * -1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * 0.0; //-4.0',
		'    fx += ((w2 * (H3 - L3)) + L3) * -2.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 0.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 0.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 2.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 0.0; //4.0',
		'    fx += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 2.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'  ',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 2.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 0.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 2.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 0.0; // 4.0',
		'    fy += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * 0.0; // -4.0',
		'    fy += ((w2 * (H7 - L7)) + L7) * -2.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 0.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * -2.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * -1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 0.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 2.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 0.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * -2.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * 0.0; //-4.0',
		'    fz += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 0.0; // 4.0',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 2.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * -1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 0.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * -2.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 0.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    */    ',
		'   ',
		'    fx =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fx += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fx += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fy += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fy += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fz += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    vec3 n = vec3( fx/27.0 , fy/27.0 , fz/27.0 );',
		'    return n;',
		'}',
		'// Compute the Normal around the current voxel',
		'vec3 getNormal(vec3 at)',
		'{',
		'    float fSliceLower, fSliceUpper, s1, s2;',
		'    float dx1, dy1, dx2, dy2;',
		'    int iTexLowerIndex, iTexUpperIndex;',
		'    vec2 texpos1,texpos2;',
		'    float slicesPerSprite = uSlicesOverX * uSlicesOverY;',
		'    fSliceLower = floor(at.z*uNumberOfSlices); // z value is between 0 and 1. Multiplying the total number of slices',
		'                                               // gives the position in between. By flooring the value, you get the lower',
		'                                               // slice position.',
		'    fSliceUpper = min(fSliceLower + 1.0, uNumberOfSlices); // return the mininimum between the two values',
		'                                                           // act as a upper clamp.',
		'    // At this point, we get our lower slice and upper slice',
		'    // Now we need to get which texture image contains our slice.',
		'    iTexLowerIndex = int(floor(fSliceLower / slicesPerSprite));',
		'    iTexUpperIndex = int(floor(fSliceUpper / slicesPerSprite));',
		'    // mod returns the value of x modulo y. This is computed as x - y * floor(x/y).',
		'    s1 = mod(fSliceLower, slicesPerSprite); // returns the index of slice in slicemap',
		'    s2 = mod(fSliceUpper, slicesPerSprite);',
		'    dx1 = fract(s1/uSlicesOverX);',
		'    dy1 = floor(s1/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    dx2 = fract(s2/uSlicesOverX);',
		'    dy2 = floor(s2/uSlicesOverY)/uSlicesOverY; // first term is the row within the slicemap',
		'                                               // second division is normalize along y-axis',
		'    float weight = at.z - floor(at.z);',
		'    float w1 = at.z - floor(at.z);',
		'    float w0 = (at.z - (1.0/uNumberOfSlices)) - floor(at.z);',
		'    float w2 = (at.z + (1.0/uNumberOfSlices)) - floor(at.z);',
		'    ',
		'    ',
		'    float fx, fy, fz;',
		'    ',
		'    float L0, L1, L2, L3, L4, L5, L6, L7, L8;',
		'    float H0, H1, H2, H3, H4, H5, H6, H7, H8;',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( iTexLowerIndex == <%=i%> )',
		'        {',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            L5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx1+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx1+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy1+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            L8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'        }',
		'        if( iTexUpperIndex == <%=i%> ) {',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H0 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H1 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H2 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H3 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H4 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y + 0.0/uNumberOfSlices)/uSlicesOverY);',
		'            H5 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x - 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H6 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        ',
		'            texpos1.x = dx2+((at.x + 0.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H7 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'            ',
		'            texpos1.x = dx2+((at.x + 1.0/uNumberOfSlices)/uSlicesOverX);',
		'            texpos1.y = dy2+((at.y - 1.0/uNumberOfSlices)/uSlicesOverY);',
		'            H8 = texture2D(uSliceMaps[<%=i%>],texpos1).x;',
		'        }',
		'    <% } %>',
		'    // we need to get interpolation of 2 x points',
		'    // x direction',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // -3 -6 -3   0  0  0   3  6  3',
		'    // -1 -3 -1   0  0  0   1  3  1',
		'    // y direction',
		'    //  1  3  1   3  6  3   1  3  1',
		'    //  0  0  0   0  0  0   0  0  0',
		'    // -1 -3 -1  -3 -6 -3  -1 -3 -1',
		'    // z direction',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    // -3  0  3   -6  0  6   -3  0  3',
		'    // -1  0  1   -3  0  3   -1  0  1',
		'    fx =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * -2.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * -1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * -4.0; //-4.0',
		'    fx += ((w2 * (H3 - L3)) + L3) * -2.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 0.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 0.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 2.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 4.0; //4.0',
		'    fx += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 2.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 2.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 0.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 2.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 4.0; // 4.0',
		'    fy += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * -4.0; // -4.0',
		'    fy += ((w2 * (H7 - L7)) + L7) * -2.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 0.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * -2.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * -1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 0.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 2.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 0.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * -2.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * -4.0; //-4.0',
		'    fz += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 4.0; // 4.0',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 2.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * -1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 0.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * -2.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 0.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    /*',
		'    fx =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * -2.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * -1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * 0.0; //-4.0',
		'    fx += ((w2 * (H3 - L3)) + L3) * -2.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 0.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 0.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 2.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 0.0; //4.0',
		'    fx += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 2.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'  ',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 2.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 0.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * -2.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * -1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 2.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 0.0; // 4.0',
		'    fy += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 0.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * 0.0; // -4.0',
		'    fy += ((w2 * (H7 - L7)) + L7) * -2.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 2.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 0.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * -2.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * -1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * -1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 0.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * -2.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 0.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 2.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * -1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 0.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * -2.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 0.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 2.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * 0.0; //-4.0',
		'    fz += ((w1 * (H4 - L4)) + L4) * 0.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 0.0; // 4.0',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * -2.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 0.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 2.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * -1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 0.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * -2.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 0.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 2.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * -1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 0.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    */    ',
		'    /*   ',
		'    fx =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fx += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fx += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fx += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fx += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fx += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fx += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fx += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fx += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fx += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fx += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fx += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fx += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fx += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fx += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fx += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fx += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fx += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fx += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fx += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fx += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fx += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fx += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fx += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fx += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    ',
		'    fy =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fy += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fy += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fy += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fy += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fy += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fy += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fy += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fy += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fy += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fy += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fy += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fy += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fy += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fy += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fy += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fy += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fy += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fy += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fy += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fy += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fy += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fy += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fy += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fy += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    fz =  ((w0 * (H0 - L0)) + L0) * 1.0;',
		'    fz += ((w1 * (H0 - L0)) + L0) * 1.0;',
		'    fz += ((w2 * (H0 - L0)) + L0) * 1.0;',
		'    ',
		'    fz += ((w0 * (H3 - L3)) + L3) * 1.0;',
		'    fz += ((w1 * (H3 - L3)) + L3) * 1.0;',
		'    fz += ((w2 * (H3 - L3)) + L3) * 1.0;',
		'    ',
		'    fz += ((w0 * (H6 - L6)) + L6) * 1.0;',
		'    fz += ((w1 * (H6 - L6)) + L6) * 1.0;',
		'    fz += ((w2 * (H6 - L6)) + L6) * 1.0;',
		'    ',
		'    fz += ((w0 * (H1 - L1)) + L1) * 1.0;',
		'    fz += ((w1 * (H1 - L1)) + L1) * 1.0;',
		'    fz += ((w2 * (H1 - L1)) + L1) * 1.0;',
		'    ',
		'    fz += ((w0 * (H4 - L4)) + L4) * 1.0;',
		'    fz += ((w1 * (H4 - L4)) + L4) * 1.0;',
		'    fz += ((w2 * (H4 - L4)) + L4) * 1.0;',
		'    ',
		'    fz += ((w0 * (H7 - L7)) + L7) * 1.0;',
		'    fz += ((w1 * (H7 - L7)) + L7) * 1.0;',
		'    fz += ((w2 * (H7 - L7)) + L7) * 1.0;',
		'    ',
		'    fz += ((w0 * (H2 - L2)) + L2) * 1.0;',
		'    fz += ((w1 * (H2 - L2)) + L2) * 1.0;',
		'    fz += ((w2 * (H2 - L2)) + L2) * 1.0;',
		'    ',
		'    fz += ((w0 * (H5 - L5)) + L5) * 1.0;',
		'    fz += ((w1 * (H5 - L5)) + L5) * 1.0;',
		'    fz += ((w2 * (H5 - L5)) + L5) * 1.0;',
		'    ',
		'    fz += ((w0 * (H8 - L8)) + L8) * 1.0;',
		'    fz += ((w1 * (H8 - L8)) + L8) * 1.0;',
		'    fz += ((w2 * (H8 - L8)) + L8) * 1.0;',
		'    */',
		'    vec3 n = vec3( fx/27.0 , fy/27.0 , fz/27.0 );',
		'    return n;',
		'}',
		'// returns intensity of reflected ambient lighting',
		'//const vec3 lightColor = vec3(1.0, 0.88, 0.74);',
		'vec3 u_intensity = vec3(0.1, 0.1, 0.1);',
		'vec3 ambientLighting(in vec3 lightColor)',
		'{',
		'    vec3 u_matAmbientReflectance = lightColor;',
		'    vec3 u_lightAmbientIntensity = u_intensity;',
		'    return u_matAmbientReflectance * u_lightAmbientIntensity;',
		'}',
		'// returns intensity of diffuse reflection',
		'vec3 diffuseLighting(in vec3 N, in vec3 L, in vec3 lightColor)',
		'{',
		'    vec3 u_matDiffuseReflectance = lightColor;',
		'    vec3 u_lightDiffuseIntensity = vec3(0.6, 0.6, 0.6);',
		'    // calculation as for Lambertian reflection',
		'    float diffuseTerm = dot(N, L);',
		'    if (diffuseTerm > 1.0) {',
		'        diffuseTerm = 1.0;',
		'    } else if (diffuseTerm < 0.0) {',
		'        diffuseTerm = 0.0;',
		'    }',
		'    return u_matDiffuseReflectance * u_lightDiffuseIntensity * diffuseTerm;',
		'}',
		'// returns intensity of specular reflection',
		'vec3 specularLighting(in vec3 N, in vec3 L, in vec3 V, in vec3 lightColor)',
		'{',
		'  float specularTerm = 0.0;',
		'    // const vec3 u_lightSpecularIntensity = vec3(0, 1, 0);',
		'    vec3 u_lightSpecularIntensity = u_intensity;',
		'    vec3 u_matSpecularReflectance = lightColor;',
		'    float u_matShininess = 5.0;',
		'   // calculate specular reflection only if',
		'   // the surface is oriented to the light source',
		'   if(dot(N, L) > 0.0)',
		'   {',
		'      vec3 e = normalize(-V);',
		'      vec3 r = normalize(-reflect(L, N));',
		'      specularTerm = pow(max(dot(r, e), 0.0), u_matShininess);',
		'   }',
		'   return u_matSpecularReflectance * u_lightSpecularIntensity * specularTerm;',
		'}',
		'void main(void) {',
		'    int uStepsI = int(uSteps);',
		'    float uStepsF = uSteps;',
		'    ',
		'    vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0; ',
		'    vec4 backColor = texture2D(uBackCoord,texC); ',
		'    vec3 dir = backColor.rgb - frontColor.rgb; ',
		'    vec4 vpos = frontColor; ',
		'    //vec3 Step = dir/uStepsF; ',
		'    vec3 Step = dir/ 256.0; ',
		'    vec4 accum = vec4(0.0, 0.0, 0.0, 0.0); ',
		'    vec4 sample = vec4(0.0, 0.0, 0.0, 0.0); ',
		'    vec4 colorValue = vec4(0.0, 0.0, 0.0, 0.0);',
		'    /*',
		'    vec3 lightPos[3] = vec3[3](',
		'        vec3(1.0, 1.0, 1.0),',
		'        vec3(-1.0, -1.0, -1.0),',
		'        vec3(1.0, 1.0, -1.0)',
		'    );',
		'    */',
		'    //vec3 lightPos[3];',
		'    //lightPos[0] = vec3(1.0, 1.0, 1.0);',
		'    //lightPos[1] = vec3(-1.0, -1.0, -1.0);',
		'    //lightPos[2] = vec3(1.0, 1.0, -1.0);',
		'    vec3 ef_position = vec3(0.0);',
		'    vec3 ef_step = vec3(0.0);',
		'    vec3 ef_value = vec3(0.0);',
		'    float opacityFactor = uOpacityVal; ',
		'    // Rasteriser for empty skipping',
		' ',
		' ',
		'    for(int i = 0; i < 8192; i++) {',
		'        if (i > uStepsI)',
		'            break;',
		'        vec3 gray_val = getVolumeValue_Soebel(vpos.xyz); ',
		'        if(gray_val.x <= uMinGrayVal ||',
		'           gray_val.x >= uMaxGrayVal)',
		'            colorValue = vec4(0.0);',
		'        else {',
		'            /*',
		'            ef_position = vpos.xyz;',
		'            ef_step = Step / 2.0;',
		'            ef_position = ef_position - ef_step;',
		'            for(int j = 0; j < 4; j++) {',
		'                ef_value = getVolume_Soebel(ef_position);',
		'                ef_step = ef_step / 2.0;',
		'                if(ef_value.x >= uMinGrayVal ||',
		'                   ef_value.x <= uMaxGrayVal) {',
		'                    // HIT',
		'                    ef_position = ef_position - ef_step;',
		'                } else {',
		'                    // NO HIT',
		'                    ef_position = ef_position + ef_step;',
		'                }',
		'            }',
		'            //float eps =pow(2.0,-16.0);',
		'            ef_position = ef_position + (1.0/900.0);',
		'            vec3 L = normalize(ef_position.xyz - uLightPos);',
		'            vec3 V = normalize( cameraPosition - ef_position.xyz );',
		'            vec3 N = normalize(getNormal(ef_position.xyz));',
		'            */',
		'            /*',
		'            // Interpolate 4 normals nearby',
		'            vec3 p0 = vpos.xyz;',
		'            p0.x -= 1.0;',
		'            p0.y += 1.0;',
		'            vec3 p1 = vpos.xyz;',
		'            p1.x -= 1.0;',
		'            p1.y -= 1.0;',
		'            vec3 p2 = vpos.xyz;',
		'            p1.x += 1.0;',
		'            p1.y += 1.0;',
		'            ',
		'            vec3 p3 = vpos.xyz;',
		'            p1.x += 1.0;',
		'            p1.y -= 1.0;',
		'            ',
		'            vec3 tmp0 = normalize(getNormal(p0.xyz));',
		'            vec3 tmp1 = normalize(getNormal(p1.xyz));',
		'            vec3 tmp2 = mix(tmp0, tmp1, 0.5);',
		'            ',
		'            vec3 tmp3 = normalize(getNormal(p2.xyz));',
		'            vec3 tmp4 = normalize(getNormal(p3.xyz));',
		'            vec3 tmp5 = mix(tmp3, tmp4, 0.5);',
		'            ',
		'            vec3 tmp6 = mix(tmp2, tmp5, 0.5);',
		'            vec3 N = tmp6;',
		'            */',
		'            /*',
		'            vec3 V = normalize(cameraPosition - vpos.xyz);',
		'            vec3 N = normalize(getNormal(vpos.xyz));',
		'            for(int light_i = 0; light_i < 1; light_i++) {',
		'              vec3 L = normalize( vpos.xyz - lightPos[light_i] );',
		'              vec3 Iamb = ambientLighting();',
		'              vec3 Idif = diffuseLighting(N, L);',
		'              vec3 Ispe = specularLighting(N, L, V);',
		'              sample.rgb += (Iamb + Idif + Ispe);',
		'            }',
		'            sample.a = 1.0;',
		'            */ ',
		'            ',
		'            // normalize vectors after interpolation',
		'            //vec3 L = normalize(vpos.xyz - uLightPos);',
		'            //vec3 L = normalize(vpos.xyz - vec3(1.0));',
		'            //vec3 L;',
		'            //for(int light_i = 0; light_i < 1; light_i++) {',
		'            //    vec3 L = normalize(vpos.xyz - vec3(0.0));',
		'            //}',
		'            //vec3 L = normalize(vpos.xyz - vec3(0.0));',
		'            colorValue = texture2D(uColormap, vec2(gray_val.x, 0.5));',
		'            ',
		'            vec3 V = normalize( cameraPosition - vpos.xyz );',
		'            //vec3 N1 = normalize(getNormal(vpos.xyz));',
		'            vec3 N = normalize(getNormalSmooth(vpos.xyz));',
		'            //vec3 N = (N1*0.3) + (N2*0.7);',
		'           ',
		'            vec3 L = normalize( vpos.xyz - vec3(1.0));',
		'            vec3 Iamb = ambientLighting(colorValue.xyz);',
		'            vec3 Idif = diffuseLighting(N, L, colorValue.xyz);',
		'            vec3 Ispe = specularLighting(N, L, V, colorValue.xyz);',
		'            sample.rgb += (Iamb + Idif + Ispe);',
		'            L = normalize( vpos.xyz - vec3(-1.0));',
		'            Iamb = ambientLighting(colorValue.xyz);',
		'            Idif = diffuseLighting(N, L, colorValue.xyz);',
		'            Ispe = specularLighting(N, L, V, colorValue.xyz);',
		'            sample.rgb += (Iamb + Idif + Ispe);',
		'           ',
		'            /* ',
		'            L = normalize( vpos.xyz - vec3(1.0, 1.0, -1.0));',
		'            Iamb = ambientLighting();',
		'            Idif = diffuseLighting(N, L);',
		'            Ispe = specularLighting(N, L, V);',
		'            sample.rgb += (Iamb + Idif + Ispe);',
		'            */',
		'            /*',
		'            // get Blinn-Phong reflectance components',
		'            vec3 Iamb = ambientLighting();',
		'            vec3 Idif = diffuseLighting(N, L);',
		'            vec3 Ispe = specularLighting(N, L, V);',
		'            // diffuse color of the object from texture',
		'            //vec3 diffuseColor = texture(u_diffuseTexture, o_texcoords).rgb;',
		'        ',
		'            vec3 mycolor = (Iamb + Idif + Ispe);',
		'            sample.rgb = mycolor;',
		'            */',
		'            sample.a = 1.0;',
		'            accum += sample; ',
		'            if(accum.a>=1.0) ',
		'               break; ',
		'        }    ',
		'   ',
		'        //advance the current position ',
		'        vpos.xyz += Step;  ',
		'   ',
		'        if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0)      ',
		'            break;  ',
		'    } ',
		'    gl_FragColor = accum; ',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassStevenNN = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uColormap" : { type: "t", value: null },
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'//see core.js -->',
		'//attributes: {',
		'//    vertColor: {type: \'c\', value: [] }',
		'//},',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'#ifdef GL_FRAGMENT_PRECISION_HIGH',
		' // highp is supported',
		' precision highp int;',
		' precision highp float;',
		'#else',
		' // high is not supported',
		' precision mediump int;',
		' precision mediump float;',
		'#endif',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uColormap;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'// returns total number of slices of all slicemaps',
		'uniform float uNumberOfSlices;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uOpacityVal;',
		'uniform float uColorVal;',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float uSlicemapWidth;',
		'float getVolumeValue(vec3 volpos)',
		'{',
		'    float value1 = 0.0;',
		'    vec2 texpos1;',
		'    vec3 value1_vec;',
		'    ',
		'    float eps =pow(2.0,-16.0);',
		'    if (volpos.x >= 1.0)',
		'        volpos.x = 1.0-eps;',
		'    if (volpos.y >= 1.0)',
		'        volpos.y = 1.0-eps;',
		'    if (volpos.z >= 1.0)',
		'        volpos.z = 1.0-eps;',
		'    ',
		'    float slicesPerSlicemap = uSlicesOverX * uSlicesOverY; ',
		'    float sliceNo = floor(volpos.z*(uNumberOfSlices));',
		'    ',
		'    int texIndexOfSlicemap = int(floor(sliceNo / slicesPerSlicemap));',
		'    float s1 = mod(sliceNo, slicesPerSlicemap);',
		'    float dx1 = fract(s1/uSlicesOverX);',
		'    float dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;      ',
		'       ',
		'    float sliceSizeX = uSlicemapWidth/uSlicesOverX;',
		'    float sliceSizeY = uSlicemapWidth/uSlicesOverY;',
		'    ',
		'    texpos1.x = dx1+(floor(volpos.x*sliceSizeX)+0.5)/uSlicemapWidth;',
		'    texpos1.y = dy1+(floor(volpos.y*sliceSizeY)+0.5)/uSlicemapWidth;',
		' ',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( texIndexOfSlicemap == <%=i%> )',
		'        {',
		'          value1_vec = texture2D(uSliceMaps[<%=i%>],texpos1).rgb;',
		'          //value1 = ((value1_vec.r + value1_vec.g + value1_vec.b)/3.0);',
		'          //value1 = ((value1_vec.r * 0.299)+(value1_vec.g * 0.587)+(value1_vec.b * 0.114));',
		'          value1 = value1_vec.r;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    ',
		'    return value1;',
		'}',
		'void main(void)',
		'{',
		' vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		' vec4 backColor = texture2D(uBackCoord,texC);',
		' vec3 dir = backColor.rgb - frontColor.rgb;',
		' vec4 vpos = frontColor;',
		' ',
		' ',
		' float dir_length = length(dir);',
		' float uStepsF = ceil((dir_length)*(uNumberOfSlices-1.0));',
		' vec3 Step = dir/(uStepsF);',
		' int uStepsI = int(uStepsF);',
		' ',
		' vec4 accum = vec4(0, 0, 0, 0);',
		' vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);',
		' vec4 colorValue = vec4(0, 0, 0, 0);',
		' float biggest_gray_value = 0.0;',
		' float opacityFactor = uOpacityVal;',
		' float lightFactor = uColorVal;',
		' ',
		' ',
		' ',
		' // Empty Skipping',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI) ',
		'         break;',
		' ',
		'     float gray_val = getVolumeValue(vpos.xyz);',
		'   ',
		'     if(gray_val <= uMinGrayVal || gray_val >= uMaxGrayVal) ',
		'         uStepsF -= 1.0;',
		'     ',
		'     vpos.xyz += Step;',
		'     ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'         break; ',
		' }',
		' vpos = frontColor;',
		' ',
		' ',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI) {',
		'         break;',
		'     }',
		'     float gray_val = getVolumeValue(vpos.xyz);',
		'     if(gray_val < uMinGrayVal || gray_val > uMaxGrayVal) {',
		'         colorValue = vec4(0.0);',
		'         accum=accum+colorValue;',
		'         if(accum.a>=1.0)',
		'            break;',
		'     } else {',
		'         // Stevens mode',
		'             vec2 tf_pos; ',
		'             tf_pos.x = (gray_val - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal); ',
		'             tf_pos.x = gray_val;',
		'             tf_pos.y = 0.5; ',
		'             colorValue = texture2D(uColormap,tf_pos);',
		'             //colorValue = texture2D(uTransferFunction,tf_pos);',
		'             //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0); ',
		'             sample.a = colorValue.a * opacityFactor * (1.0 / uStepsF); ',
		'             //sample.rgb = (1.0 - accum.a) * colorValue.rgb * sample.a * uColorVal; ',
		'             sample.rgb = colorValue.rgb; ',
		'             accum += sample; ',
		'             if(accum.a>=1.0) ',
		'                break; ',
		'     }',
		'     //advance the current position',
		'     vpos.xyz += Step;',
		'     ',
		'     //break if the position is greater than <1, 1, 1> ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'     { ',
		'         break; ',
		'     } ',
		'     ',
		' }',
		' gl_FragColor = accum;',
		'}'].join("\n")
};
window.VRC.Core.prototype._shaders.secondPassStevenTri = {
	uniforms: THREE.UniformsUtils.merge([
		{
		"uBackCoord" : { type: "t", value: null },
		"uTransferFunction" : { type: "t", value: null },
		"uSliceMaps" : { type: "tv", value: [] },
		"uNumberOfSlices" : { type: "f", value: -1 },
		"uMinGrayVal" : { type: "f", value: -1 },
		"uMaxGrayVal" : { type: "f", value: -1 },
		"uOpacityVal" : { type: "f", value: -1 },
		"uColorVal" : { type: "f", value: -1 },
		"uAbsorptionModeIndex" : { type: "f", value: -1 },
		"uSlicesOverX" : { type: "f", value: -1 },
		"uSlicesOverY" : { type: "f", value: -1 },
		"uSlicemapWidth" : { type: "f", value: -1 },
		}
	]),
	vertexShader: [
		'precision mediump int;',
		'precision mediump float;',
		'attribute vec4 vertColor;',
		'//see core.js -->',
		'//attributes: {',
		'//    vertColor: {type: \'c\', value: [] }',
		'//},',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'void main(void)',
		'{',
		'    frontColor = vertColor;',
		'    pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
		'    gl_Position = pos;',
		'}'].join("\n"),
	fragmentShader: [
		'#ifdef GL_FRAGMENT_PRECISION_HIGH',
		' // highp is supported',
		' precision highp int;',
		' precision highp float;',
		'#else',
		' // high is not supported',
		' precision mediump int;',
		' precision mediump float;',
		'#endif',
		'varying vec4 frontColor;',
		'varying vec4 pos;',
		'uniform sampler2D uBackCoord;',
		'uniform sampler2D uTransferFunction;',
		'uniform sampler2D uSliceMaps[<%= maxTexturesNumber %>];',
		'// returns total number of slices of all slicemaps',
		'uniform float uNumberOfSlices;',
		'uniform float uMinGrayVal;',
		'uniform float uMaxGrayVal;',
		'uniform float uOpacityVal;',
		'uniform float uColorVal;',
		'uniform float uAbsorptionModeIndex;',
		'uniform float uSlicesOverX;',
		'uniform float uSlicesOverY;',
		'uniform float uSlicemapWidth;',
		'float getTextureValue(int slicemapNo, vec2 texpos)',
		'{',
		'    float value = 0.0;',
		'    vec3 value_vec;',
		'    ',
		'    <% for(var i=0; i < maxTexturesNumber; i++) { %>',
		'        if( slicemapNo == <%=i%> )',
		'        {',
		'          value_vec = texture2D(uSliceMaps[<%=i%>],texpos).rgb;',
		'          //value = ((value_vec.r + value_vec.g + value_vec.b)/3.0);',
		'          value = ((value_vec.r * 0.299)+(value_vec.g * 0.587)+(value_vec.b * 0.114));',
		'          value = vale_rec.r;',
		'        }',
		'        <% if( i < maxTexturesNumber-1 ) { %>',
		'            else',
		'        <% } %>',
		'    <% } %>',
		'    ',
		'    return value;',
		'}',
		'float getValueTri(vec3 volpos)',
		'{',
		'    vec2 texpos1a, texpos1b, texpos1c, texpos1d, texpos2a, texpos2b, texpos2c, texpos2d;',
		'    float value1a, value1b, value1c, value1d, value2a, value2b, value2c, value2d, valueS;',
		'    float value1ab, value1cd, value1ac, value1bd, value2ab, value2cd, value2ac, value2bd, value1, value2;',
		'    float NOS = uNumberOfSlices;  //  abbreviation ',
		'    float slicesPerSlicemap = uSlicesOverX * uSlicesOverY; ',
		'    float sliceSizeX = uSlicemapWidth/uSlicesOverX;  // Number of pixels of ONE slice along x axis',
		'    float sliceSizeY = uSlicemapWidth/uSlicesOverY;  // Number of pixels of ONE slice along y axis',
		'    ',
		'    //  Slice selection',
		'    float sliceNo1 = floor(abs(volpos.z*NOS-0.5));  //  sliceNo1 stands for lower slice',
		'    float sliceNo2 = NOS-1.0-floor(abs(NOS-0.5-volpos.z*NOS));  //  sliceNo2 stands for upper slice',
		'    int slicemapNo1 = int(floor(sliceNo1 / slicesPerSlicemap));',
		'    int slicemapNo2 = int(floor(sliceNo2 / slicesPerSlicemap));',
		'    float s1 = mod(sliceNo1, slicesPerSlicemap);  // s1 stands for the sliceNo of lower slice in this map',
		'    float dx1 = fract(s1/uSlicesOverX);',
		'    float dy1 = floor(s1/uSlicesOverY)/uSlicesOverY;',
		'    float s2 = mod(sliceNo2, slicesPerSlicemap);  // s2 stands for the sliceNo of upper slice in this map',
		'    float dx2 = fract(s2/uSlicesOverX);',
		'    float dy2 = floor(s2/uSlicesOverY)/uSlicesOverY;',
		'    ',
		'    /*',
		'    texpos1.x = dx1+volpos.x/uSlicesOverX;  // directly from texture2D',
		'    texpos1.y = dy1+volpos.y/uSlicesOverY;',
		'    texpos1.x = dx1+(floor(volpos.x*sliceSizeX)+0.5)/uSlicemapWidth;  //  NearestNeighbor in lower slice',
		'    texpos1.y = dy1+(floor(volpos.y*sliceSizeY)+0.5)/uSlicemapWidth;',
		'    */',
		'    ',
		'    // Four nearest pixels in lower slice',
		'    texpos1a.x = texpos1c.x = dx1+(floor(abs(volpos.x*sliceSizeX-0.5))+0.5)/uSlicemapWidth;  //  Trilinear',
		'    texpos1a.y = texpos1b.y = dy1+(floor(abs(volpos.y*sliceSizeY-0.5))+0.5)/uSlicemapWidth;',
		'    texpos1b.x = texpos1d.x = dx1+(sliceSizeX-1.0-floor(abs(sliceSizeX-0.5-volpos.x*sliceSizeX))+0.5)/uSlicemapWidth;',
		'    texpos1c.y = texpos1d.y = dy1+(sliceSizeY-1.0-floor(abs(sliceSizeY-0.5-volpos.y*sliceSizeY))+0.5)/uSlicemapWidth;',
		'    ',
		'    // Four nearest pixels in upper slice',
		'    texpos2a.x = texpos2c.x = dx2+(floor(abs(volpos.x*sliceSizeX-0.5))+0.5)/uSlicemapWidth;  //  Trilinear',
		'    texpos2a.y = texpos2b.y = dy2+(floor(abs(volpos.y*sliceSizeY-0.5))+0.5)/uSlicemapWidth;',
		'    texpos2b.x = texpos2d.x = dx2+(sliceSizeX-1.0-floor(abs(sliceSizeX-0.5-volpos.x*sliceSizeX))+0.5)/uSlicemapWidth;',
		'    texpos2c.y = texpos2d.y = dy2+(sliceSizeY-1.0-floor(abs(sliceSizeY-0.5-volpos.y*sliceSizeY))+0.5)/uSlicemapWidth;',
		'    // get texture values of these 8 pixels',
		'    value1a = getTextureValue(slicemapNo1, texpos1a);',
		'    value1b = getTextureValue(slicemapNo1, texpos1b);',
		'    value1c = getTextureValue(slicemapNo1, texpos1c);',
		'    value1d = getTextureValue(slicemapNo1, texpos1d);',
		'    value2a = getTextureValue(slicemapNo2, texpos2a);',
		'    value2b = getTextureValue(slicemapNo2, texpos2b);',
		'    value2c = getTextureValue(slicemapNo2, texpos2c);',
		'    value2d = getTextureValue(slicemapNo2, texpos2d);',
		'    ',
		'    // ratio calculation',
		'    float ratioX = volpos.x*sliceSizeX+0.5-floor(volpos.x*sliceSizeX+0.5);',
		'    float ratioY = volpos.y*sliceSizeY+0.5-floor(volpos.y*sliceSizeY+0.5);',
		'    float ratioZ = volpos.z*NOS+0.5-floor(volpos.z*NOS+0.5);',
		'    //float ratioZ = (volpos.z-(sliceNo1+0.5)/NOS) / (1.0/NOS);  // Another way to get ratioZ',
		'    ',
		'    ',
		'    //  Trilinear interpolation ',
		'    value1ab = value1a+ratioX*(value1b-value1a);',
		'    value1cd = value1c+ratioX*(value1d-value1c);',
		'    value1 = value1ab+ratioY*(value1cd-value1ab);',
		'    value2ab = value2a+ratioX*(value2b-value2a);',
		'    value2cd = value2c+ratioX*(value2d-value2c);',
		'    value2 = value2ab+ratioY*(value2cd-value2ab);',
		'    ',
		'    valueS = value1+ratioZ*(value2-value1);',
		'    ',
		'    ',
		'    // Do NO interpolation with empty voxels',
		'    if (value1a<=0.0 || value1b<=0.0 || value1c<=0.0 || value1d<=0.0 || value2a<=0.0 || value2b<=0.0 || value2c<=0.0 || value2d<=0.0)',
		'    {',
		'        if (value1a<=0.0 || value1c<=0.0 || value2a<=0.0 || value2c<=0.0)',
		'        {    ',
		'            value1ab = value1b;',
		'            value1cd = value1d;',
		'            value2ab = value2b;',
		'            value2cd = value2d;',
		'            ',
		'            if (value1b<=0.0 || value2b<=0.0)',
		'            {',
		'                value1 = value1d;',
		'                value2 = value2d;',
		'                ',
		'                if (value1d <= 0.0)',
		'                    valueS = value2;',
		'                else if (value2d <= 0.0)',
		'                    valueS = value1;',
		'                else',
		'                    valueS = value1+ratioZ*(value2-value1);',
		'            }',
		'            ',
		'            else if (value1d<=0.0 || value2d<=0.0)',
		'            {',
		'                value1 = value1b;',
		'                value2 = value2b;',
		'                valueS = value1+ratioZ*(value2-value1);',
		'            }',
		'            ',
		'            else',
		'            {',
		'                value1 = value1ab+ratioY*(value1cd-value1ab);',
		'                value2 = value2ab+ratioY*(value2cd-value2ab);',
		'                valueS = value1+ratioZ*(value2-value1);',
		'            }',
		'        }',
		'    ',
		'    ',
		'        else',
		'        {  // if (value1b<=0.0 || value1d<=0.0 || value2b<=0.0 || value2d<=0.0)',
		'            value1ab = value1a;',
		'            value1cd = value1c;',
		'            value2ab = value2a;',
		'            value2cd = value2c;',
		'            ',
		'            value1 = value1ab+ratioY*(value1cd-value1ab);',
		'            value2 = value2ab+ratioY*(value2cd-value2ab);',
		'            valueS = value1+ratioZ*(value2-value1);',
		'        }',
		'    ',
		'    }',
		'    ',
		'    ',
		'    /*',
		'    if (value1a<=0.0 || value1b<=0.0 || value1c<=0.0 || value1d<=0.0 || value2a<=0.0 || value2b<=0.0 || value2c<=0.0 || value2d<=0.0)',
		'        valueS = 0.0;',
		'    */',
		'    ',
		'    return valueS;',
		'}',
		'void main(void)',
		'{',
		' vec2 texC = ((pos.xy/pos.w) + 1.0) / 2.0;',
		' vec4 backColor = texture2D(uBackCoord,texC);',
		' vec3 dir = backColor.rgb - frontColor.rgb;',
		' vec4 vpos = frontColor;',
		' ',
		' ',
		' float dir_length = length(dir);',
		' float uStepsF = ceil((dir_length)*(uNumberOfSlices-1.0));',
		' vec3 Step = dir/(uStepsF);',
		' int uStepsI = int(uStepsF);',
		' ',
		' vec4 accum = vec4(0, 0, 0, 0);',
		' vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);',
		' vec4 colorValue = vec4(0, 0, 0, 0);',
		' float biggest_gray_value = 0.0;',
		' float opacityFactor = uOpacityVal;',
		' float lightFactor = uColorVal;',
		' ',
		' ',
		' ',
		' // Empty Skipping',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI) ',
		'         break;',
		' ',
		'     float gray_val = getValueTri(vpos.xyz);',
		'   ',
		'     if(gray_val <= uMinGrayVal || gray_val >= uMaxGrayVal) ',
		'         uStepsF -= 1.0;',
		'     ',
		'     vpos.xyz += Step;',
		'     ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'         break; ',
		' }',
		' vpos = frontColor;',
		' ',
		' ',
		' ',
		' for(int i = 0; i < 4096; i+=1)',
		' {',
		'     if(i == uStepsI)',
		'         break;',
		'     float gray_val = getValueTri(vpos.xyz);',
		'     if(gray_val < uMinGrayVal || gray_val > uMaxGrayVal) ',
		'     {',
		'         colorValue = vec4(0.0);',
		'         accum=accum+colorValue;',
		'         if(accum.a>=1.0)',
		'            break;',
		'     } ',
		'     else ',
		'     {',
		'         if(biggest_gray_value < gray_val) ',
		'            biggest_gray_value = gray_val;',
		'         if(uAbsorptionModeIndex == 0.0)',
		'         {',
		'           vec2 tf_pos;',
		'           //tf_pos.x = (gray_val - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal);',
		'           tf_pos.x = gray_val;',
		'           tf_pos.y = 0.5;',
		'           colorValue = texture2D(uTransferFunction,tf_pos);',
		'           //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0);',
		'           sample.a = 1.0; ',
		'           //sample.a = colorValue.a * opacityFactor;',
		'           sample.rgb = colorValue.rgb * lightFactor;',
		'           accum += sample;',
		'           if(accum.a>=1.0)',
		'              break;',
		'         }',
		'         ',
		'         /*',
		'         // Guevens mode',
		'         if(uAbsorptionModeIndex == 1.0)',
		'         {',
		'           vec2 tf_pos;',
		'           ',
		'           //tf_pos.x = (gray_val - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal);',
		'           // position of x is defined by the gray_value instead of a filtering',
		'           tf_pos.x = gray_val;',
		'           tf_pos.y = 0.5;',
		'           // maximum distance in a cube',
		'           float max_d = sqrt(3.0);',
		'           colorValue = texture2D(uTransferFunction,tf_pos);',
		'           //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0);',
		'           // alternative mode, this way the user can change the length of',
		'           // the penetrating ray, by using the opacityFactor-switch in the  gui',
		'           // -2.0 because of 1. and last slice:',
		'           sample.a = 1.0/(1.0+uOpacityVal*(max_d*(uNumberOfSlices-2.0)));',
		'           sample.rgb = (1.0 - accum.a) * colorValue.rgb * sample.a * lightFactor; ',
		'           //sample.rgb =  colorValue.rgb * sample.a;',
		'           accum += sample;',
		'           if(accum.a>=1.0)',
		'              break;',
		'         }*/',
		'         ',
		'         ',
		'         // Stevens mode',
		'         if(uAbsorptionModeIndex == 1.0) ',
		'         { ',
		'             vec2 tf_pos; ',
		'             //tf_pos.x = (gray_val - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal); ',
		'             tf_pos.x = gray_val;',
		'             tf_pos.y = 0.5; ',
		'             colorValue = texture2D(uTransferFunction,tf_pos);',
		'             //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0); ',
		'             sample.a = colorValue.a * opacityFactor * (1.0 / uStepsF); ',
		'             sample.rgb = (1.0 - accum.a) * colorValue.rgb * sample.a * lightFactor; ',
		'             accum += sample; ',
		'             if(accum.a>=1.0) ',
		'                break; ',
		'                ',
		'         }',
		'         ',
		'         ',
		'         if(uAbsorptionModeIndex == 2.0)',
		'         {',
		'             vec2 tf_pos;',
		'             //tf_pos.x = (biggest_gray_value - uMinGrayVal) / (uMaxGrayVal - uMinGrayVal);',
		'             tf_pos.x = biggest_gray_value;',
		'             tf_pos.y = 0.5;',
		'             colorValue = texture2D(uTransferFunction,tf_pos);',
		'             //colorValue = vec4(tf_pos.x, tf_pos.x, tf_pos.x, 1.0);',
		'             sample.a = 1.0; //colorValue.a * opacityFactor;',
		'             sample.rgb = colorValue.rgb * lightFactor;',
		'             accum = sample;',
		'         }',
		'     }',
		'     //advance the current position',
		'     vpos.xyz += Step;',
		'     ',
		'     //break if the position is greater than <1, 1, 1> ',
		'     if(vpos.x > 1.0 || vpos.y > 1.0 || vpos.z > 1.0 || vpos.x < 0.0 || vpos.y < 0.0 || vpos.z < 0.0) ',
		'         break; ',
		'     ',
		' }',
		' gl_FragColor = accum;',
		'}'].join("\n")
};

(function(namespace) {
    var VolumeRaycaster = function(config) {

        var me = {};

        me._token;
        me._token_array = [];

        me._needRedraw = true;

        me._isStart = false;
        me._isChange = false;

        me._clock = new THREE.Clock();

        me._onLoadSlicemap              = new VRC.EventDispatcher();
        me._onLoadSlicemaps             = new VRC.EventDispatcher();

        me._core = new VRC.Core( config );
        me._adaptationManager = new VRC.AdaptationManager();

        me.init = function() {
            me._core.init();
            me._adaptationManager.init( me._core );

            var frames = 0;

            me.addCallback("onCameraChange", function() {
                me._needRedraw = true;
                me.isChange = true;
            });

            me.addCallback("onCameraChangeStart", function() {
                me.setRenderSize(me._core.getRenderSizeDefault()[0])
                me._needRedraw = true;
                me.isChange = true;
                clearInterval(me._token);
                for (i = 0; i < me._token_array.length; i++) {
                    clearInterval(me._token_array[i]);
                }
                me._token_array = [];
            });

            me.addCallback("onCameraChangeEnd", function() {
                me._token = setInterval(function(){
                    me.setRenderSize(me._core.getRenderSizeDefault()[1])
                    me._needRedraw = true;
                    me.isChange = true;
                    console.log("WAVE: stop()");
                    for (i = 0; i < me._token_array.length; i++) {
                        clearInterval(me._token_array[i]);
                    }
                    me._token_array = [];
                }, 1000);
                me._token_array.push(me._token);
            });


            var counter = 0;

            function animate() {

                requestAnimationFrame( animate );
                // Note: isStart is a flag to indicate texture maps finished loaded.
                if(me._needRedraw && me._isStart) {
                    me._core.draw(0);
                }
            };

            animate();

        };

        me.setSlicemapsImages = function(images, imagesPaths) {
            var maxTexSize = me._core.getMaxTextureSize();
            var maxTexturesNumber = me._core.getMaxTexturesNumber();

            var imagesNumber = images.length;

            if( imagesNumber > maxTexturesNumber ) {
                throw Error("Number of slicemaps bigger then number of available texture units. Available texture units: " + maxTexturesNumber);
            };

            me._core.setSlicemapsImages(imagesPaths);
            me._needRedraw = true;
        };

        me.uploadSlicemapsImages = function(imagesPaths, userOnLoadImage, userOnLoadImages, userOnError) {

            var downloadImages = function(imagesPaths, onLoadImage, onLoadImages, onError) {
                var downloadedImages = [];
                var downloadedImagesNumber = 0;

                try {
                    for (var imageIndex = 0; imageIndex < imagesPaths.length; imageIndex++) {
                        var image = new Image();
                        (function(image, imageIndex) {
                            image.onload = function() {
                                downloadedImages[imageIndex] = image;
                                downloadedImagesNumber++;

                                onLoadImage(image);

                                if(downloadedImagesNumber == imagesPaths.length) {
                                    onLoadImages(downloadedImages);
                                };
                            };
                            image.onerror = onError;
                            image.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";
                        })(image, imageIndex);

                    };
                }
                catch(e) {
                    onError(e);
                };
            };
            downloadImages(imagesPaths,
                function(image) {
                    // downloaded one of the images
                    me._onLoadSlicemap.call(image);
                    if(userOnLoadImage != undefined) userOnLoadImage(image);
                },
                function(images) {
                    // downloaded all images
                    me.setSlicemapsImages(images, imagesPaths);
                    me.start();

                    me._onLoadSlicemaps.call(images);

                    if(userOnLoadImages != undefined) userOnLoadImages(images);

                },
                function(error) {
                    // error appears
                    if(userOnError != undefined) {
                        userOnError(error);
                    } else {
                        console.error(error);

                    }
                }
            )

        };

        me.start = function() {
            me._isStart = true;
            console.log("WAVE: start()");
        };

        me.stop = function() {
            me._isStart = false;
            console.log("WAVE: stop()");
        };

        me.setSteps = function(steps_number) {
            if( steps_number <= me._core.getMaxStepsNumber() ) {
                me._core.setSteps(steps_number);
                me._needRedraw = true;

            } else {
                throw Error("Number of steps should be lower of equal length of min volume dimension.");

            }

        };

        me.setAutoStepsOn = function(flag) {
            me._adaptationManager.run(flag);
            me._needRedraw = true;

        };

        me.setSlicesRange = function(from, to) {
            me._core.setSlicesRange(from, to);
            me._needRedraw = true;

        };

        me.setMode = function(conf){
          me._core.setMode(conf);
          me._needRedraw = true;
        };

        me.setShaderName = function(value) {
          me._core.setShaderName(value);
          me._needRedraw = true;
        };
        
        me.setShader = function(codeblock){
          me._core.setShader(codeblock);
          me._needRedraw = true;
        };

        me.setShader = function(codeblock){
          me._core.setShader(codeblock);
          me._needRedraw = true;
        };

        me.setZoom = function(x1, x2, y1, y2){
          me._core.setZoom(x1, x2, y1, y2);
          me._needRedraw = true;
        };

        me.setOpacityFactor = function(opacity_factor) {
            me._core.setOpacityFactor(opacity_factor);
            me._needRedraw = true;

        };

        me.setColorFactor = function(color_factor) {
            me._core.setColorFactor(color_factor);
            me._needRedraw = true;

        };

        me.setAbsorptionMode = function(mode_index) {
            me._core.setAbsorptionMode(mode_index);
            me._needRedraw = true;

        };

        me.setIndexOfImage = function(indexOfImage) {

            me._core.setIndexOfImage(indexOfImage);
            me._needRedraw = true;


        };

        me.setVolumeSize = function(width, height, depth) {
            me._core.setVolumeSize(width, height, depth);
            me._needRedraw = true;

        };

        me.setGeometryMinX = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value > me._core.getGeometryDimensions()["xmax"]) {
                throw Error("Min X should be lower than max X!");
            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["xmin"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setGeometryMaxX = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value < me._core.getGeometryDimensions()["xmin"]) {
                throw Error("Max X should be bigger than min X!");
            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["xmax"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setGeometryMinY = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value > me._core.getGeometryDimensions()["ymax"]) {
                throw Error("Min Y should be lower than max Y!");
            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["ymin"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setGeometryMaxY = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value < me._core.getGeometryDimensions()["ymin"]) {
                throw Error("Max Y should be bigger than min Y!");

            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["ymax"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setGeometryMinZ = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value > me._core.getGeometryDimensions()["zmax"]) {
                throw Error("Min Z should be lower than max Z!");
            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["zmin"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setGeometryMaxZ = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Geometry size  should be in range [0.0 - 1.0] !");
            }

            if(value < me._core.getGeometryDimensions()["zmin"]) {
                throw Error("Max Z should be bigger than min Z!");
            }

            var geometryDimension = me._core.getGeometryDimensions();
            geometryDimension["zmax"] = value;

            me._core.setGeometryDimensions(geometryDimension);
            me._needRedraw = true;
        };

        me.setRenderCanvasSize = function(width, height) {
            me._core.setRenderCanvasSize(width, height);
            me._needRedraw = true;

        };

        me.showISO = function() {
            return me._core.showISO();
        };

        me.showVolren = function() {
            return me._core.showVolren();
        };

        me.setAxis = function() {
            me._core.setAxis();
            me._needRedraw = true;
        };

        me.removeWireframe = function() {
            me._core.removeWireframe();
            me._needRedraw = true;
        };

        me.addWireframe = function() {
            me._core.addWireframe();
            me._needRedraw = true;
        };

        me.setBackgroundColor = function(color) {
            me._core.setBackgroundColor(color);
            me._needRedraw = true;

        };

        me.setRowCol = function(row, col) {
            me._core.setRowCol(row, col);
            me._needRedraw = true;

        };
        
        me.setZoomColor = function(value) {
            me._core.setZoomColor(value);
            me._needRedraw = true;
        };
        
        me.setZoomXMinValue = function(value) {
            me._core.setZoomXMinValue(value);
            me._needRedraw = true;
        };
        
        me.setZoomXMaxValue = function(value) {
            me._core.setZoomXMaxValue(value);
            me._needRedraw = true;
        };
        
        me.setZoomYMinValue = function(value) {
            me._core.setZoomYMinValue(value);
            me._needRedraw = true;
        };
        
        me.setZoomYMaxValue = function(value) {
            me._core.setZoomYMaxValue(value);
            me._needRedraw = true;
        };
        
        me.setZoomZMinValue = function(value) {
            me._core.setZoomZMinValue(value);
            me._needRedraw = true;
        };
        
        me.setZoomZMaxValue = function(value) {
            me._core.setZoomZMaxValue(value);
            me._needRedraw = true;
        };

        me.showZoomBox = function(value) {
            me._core.showZoomBox(value);
            me._needRedraw = true;
        };
        
        me.setGrayMinValue = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Gray value should be in range [0.0 - 1.0] !");
            }

            if(value > me.getGrayMaxValue()) {
                throw Error("Gray min value should be lower than max value!");
            }

            me._core.setGrayMinValue(value);
            me._needRedraw = true;

        };

        me.setGrayMaxValue = function(value) {
            if(value > 1.0 || value < 0.0) {
                throw Error("Gray value should be in range [0.0 - 1.0] !");
            }

            if(value < me.getGrayMinValue()) {
                throw Error("Gray max value should be bigger than min value!");
            }

            me._core.setGrayMaxValue(value);
            me._needRedraw = true;

        };

        me.setTransferFunctionByColors = function(colors) {
            me._core.setTransferFunctionByColors(colors);
            me._needRedraw = true;

        };

        me.setTransferFunctionByImage = function(image) {
            me._core.setTransferFunctionByImage(image);
            me._needRedraw = true;

        };

        me.addCallback = function(event_name, callback, needStart) {
            switch(event_name) {
                case "onPreDraw": return me._core.onPreDraw.add(callback, needStart);
                case "onPostDraw": return me._core.onPostDraw.add(callback, needStart);
                case "onResizeWindow": return me._core.onResizeWindow.add(callback, needStart);
                case "onCameraChange": return me._core.onCameraChange.add(callback, needStart);
                case "onCameraChangeStart": return me._core.onCameraChangeStart.add(callback, needStart);
                case "onCameraChangeEnd": return me._core.onCameraChangeEnd.add(callback, needStart);
                case "onChangeTransferFunction": return me._core.onChangeTransferFunction.add(callback, needStart);
                case "onLoadSlicemap": return me._onLoadSlicemap.add(callback, needStart);
                case "onLoadSlicemaps": return me._onLoadSlicemaps.add(callback, needStart);
            }
            me._needRedraw = true;

        };

        me.removeCallback = function(event_name, index) {
            switch(event_name) {
                case "onPreDraw": return me._core.onPreDraw.remove(index);
                case "onPostDraw": return me._core.onPostDraw.remove(index);
                case "onResizeWindow": return me._core.onResizeWindow.remove(index);
                case "onCameraChange": return me._core.onCameraChange.remove(index);
                case "onCameraChangeStart": return me._core.onCameraChangeStart.remove(index);
                case "onCameraChangeEnd": return me._core.onCameraChangeEnd.remove(index);
                case "onChangeTransferFunction": return me._core.onChangeTransferFunction.remove(index);
                case "onLoadSlicemap": return me._onLoadSlicemap.remove(callback, needStart);
                case "onLoadSlicemaps": return me._onLoadSlicemaps.remove(callback, needStart);

            }
            me._needRedraw = true;

        };

        me.startCallback = function(event_name, index) {
            switch(event_name) {
                case "onPreDraw": return me._core.onPreDraw.start(index);
                case "onPostDraw": return me._core.onPostDraw.start(index);
                case "onResizeWindow": return me._core.onResizeWindow.start(index);
                case "onCameraChange": return me._core.onCameraChange.start(index);
                case "onCameraChangeStart": return me._core.onCameraChangeStart.start(index);
                case "onCameraChangeEnd": return me._core.onCameraChangeEnd.start(index);
                case "onChangeTransferFunction": return me._core.onChangeTransferFunction.start(index);
                case "onLoadSlicemap": return me._onLoadSlicemap.start(callback, needStart);
                case "onLoadSlicemaps": return me._onLoadSlicemaps.start(callback, needStart);

            }
            me._needRedraw = true;

        };

        me.stopCallback = function(event_name, index) {
            switch(event_name) {
                case "onPreDraw": return me._core.onPreDraw.stop(index);
                case "onPostDraw": return me._core.onPostDraw.stop(index);
                case "onResizeWindow": return me._core.onResizeWindow.stop(index);
                case "onCameraChange": return me._core.onCameraChange.stop(index);
                case "onCameraChangeStart": return me._core.onCameraChangeStart.stop(index);
                case "onCameraChangeEnd": return me._core.onCameraChangeEnd.stop(index);
                case "onChangeTransferFunction": return me._core.onChangeTransferFunction.stop(index);
                case "onLoadSlicemap": return me._onLoadSlicemap.stop(callback, needStart);
                case "onLoadSlicemaps": return me._onLoadSlicemaps.stop(callback, needStart);

            }
            me._needRedraw = true;

        };

        me.isStartCallback = function(event_name, index) {
            switch(event_name) {
                case "onPreDraw": return me._core.onPreDraw.isStart(index);
                case "onPostDraw": return me._core.onPostDraw.isStart(index);
                case "onResizeWindow": return me._core.onResizeWindow.isStart(index);
                case "onCameraChange": return me._core.onCameraChange.isStart(index);
                case "onCameraChangeStart": return me._core.onCameraChangeStart.isStart(index);
                case "onCameraChangeEnd": return me._core.onCameraChangeEnd.isStart(index);
                case "onChangeTransferFunction": return me._core.onChangeTransferFunction.isStart(index);
                case "onLoadSlicemap": return me._onLoadSlicemap.isStart(callback, needStart);
                case "onLoadSlicemaps": return me._onLoadSlicemaps.isStart(callback, needStart);

            }
            me._needRedraw = true;

        };

        me.version = function() {
            return me._core.getVersion();
        };

        me.setRenderSize = function(value) {
            return me._core.setRenderSize(value);
        };

        me.setRenderSizeDefault = function(value) {
            return me._core.setRenderSizeDefault(value);
        };

        me.getGrayMaxValue = function() {
            return me._core.getGrayMaxValue();
        };

        me.getGrayMinValue = function() {
            return me._core.getGrayMinValue();
        };

        me.getSteps = function() {
            return me._core.getSteps();
        };

        me.getSlicesRange = function() {
            return me._core.getSlicesRange();
        };

        me.getRowCol = function() {
            return me._core.getRowCol();
        };

        me.getGrayValue = function() {
            return [me._core.getGrayMinValue(), me._core.getGrayMaxValue()]
        };

        me.getGeometryDimensions = function() {
            return me._core.getGeometryDimensions();
        };

        me.getVolumeSize = function() {
            return me._core.getVolumeSize();
        };

        me.getVolumeSizeNormalized = function() {
            return me._core.getVolumeSizeNormalized();
        };

        me.getMaxStepsNumber = function() {
            return me._core.getMaxStepsNumber();
        };

        me.getMaxTextureSize = function() {
            return me._core.getMaxTextureSize();
        };

        me.getMaxTexturesNumber = function() {
            return me._core.getMaxTexturesNumber();
        };

        me.getMaxFramebuferSize = function() {
            return me._core.getMaxFramebuferSize();
        };

        me.getOpacityFactor = function() {
            return me._core.getOpacityFactor();
        };

        me.getColorFactor = function() {
            return me._core.getColorFactor();
        };

        me.getBackground = function() {
            return me._core.getBackground();
        };

        me.getAbsorptionMode = function() {
            return me._core.getAbsorptionMode();
        };

        me.getRenderSize = function() {
            return me._core.getRenderSize();
        };

        me.getRenderSizeInPixels  = function() {
            return me._core.getRenderSizeInPixels();
        };

        me.getRenderCanvasSize = function() {
            return me._core.getCanvasSize();
        };

        me.getRenderCavnvasSizeInPixels  = function() {
            return me._core.getCanvasSizeInPixels();
        };

        me.getAbsorptionMode = function() {
            return me._core.getAbsorptionMode();
        };

        me.getSlicemapsPaths = function() {
            return me._core.getSlicemapsPaths();
        };

        me.getDomContainerId = function() {
            return me._core.getDomContainerId();
        };

        me.getCameraSettings = function() {
            return me._core.getCameraSettings();
        };

        me.getGeometrySettings = function() {
            return me._core.getGeometrySettings();
        };

        me.getDomContainerId = function() {
            return me._core.getDomContainerId();
        };

        me.getClearColor = function() {
            return me._core.getClearColor();
        };

        me.getTransferFunctionColors = function() {
            return me._core.getTransferFunctionColors();
        };

        me.getTransferFunctionAsImage = function() {
            return me._core.getTransferFunctionAsImage();
        };

        me.getBase64 = function() {
            return me._core.getBase64();
        };

        me.set2DTexture = function(urls) {
            me._core.set2DTexture(urls);
            me._needRedraw = true;
            return true;
        };

        me.isAutoStepsOn = function() {
            console.log("Check");
            console.log(me._adaptationManager.isRun());
            return me._adaptationManager.isRun();
        };

        me.setAxis = function() {
            return me._core.setAxis();
        };

        me.draw = function() {
            me._core.draw();
        };

        me.setConfig = function(config, onLoadImage, onLoadImages) {
            if(config['slicemaps_images'] != undefined) {
                me.setSlicemapsImages( config['slicemaps_images'] );
            }

            if(config['slicemaps_paths'] != undefined) {
                me.uploadSlicemapsImages(

                    config['slicemaps_paths'],
                    function(image) {
                        if(onLoadImage != undefined) onLoadImage(image);
                    },
                    function(images) {
                        if(config['slices_range'] != undefined) {
                            me.setSlicesRange( config['slices_range'][0], config['slices_range'][1] );
                        }
                        me.stop();
                        if(onLoadImages != undefined) onLoadImages(images);

                        me.start();
                    }
                );
            }

            if(config['slices_range'] != undefined) {
                me.setSlicesRange( config['slices_range'][0], config['slices_range'][1] );
            }

            if(config['steps'] != undefined) {
                me._core.setSteps( config['steps'] );
            }

            if(config['row_col'] != undefined) {
                me._core.setRowCol( config['row_col'][0], config['row_col'][1] );
            }

            if(config['gray_min'] != undefined) {
                me._core.setGrayMinValue( config['gray_min'] );
            }

            if(config['gray_max'] != undefined) {
                me._core.setGrayMaxValue( config['gray_max'] );
            }

            if(config['threshold_indexes'] != undefined) {
                me._core.setThresholdIndexes( config['threshold_indexes']["otsu"], config['threshold_indexes']["isodata"], config['threshold_indexes']["yen"], config['threshold_indexes']["li"] );
            }

            if(config['volume_size'] != undefined) {
                me.setVolumeSize( config['volume_size'][0], config['volume_size'][1], config['volume_size'][2] );
            }

            if(config['x_min'] != undefined) {
                me.setGeometryMinX( config['x_min'] );
            }

            if(config['x_max'] != undefined) {
                me.setGeometryMaxX( config['x_max'] );
            }

            if(config['y_min'] != undefined) {
                me.setGeometryMinY( config['y_min'] );
            }

            if(config['y_max'] != undefined) {
                me.setGeometryMaxY( config['y_max'] );
            }

            if(config['z_min'] != undefined) {
                me.setGeometryMinZ( config['z_min'] );
            }

            if(config['z_max'] != undefined) {
                me.setGeometryMaxZ( config['z_max'] );
            }

            if(config['opacity_factor'] != undefined) {
                me._core.setOpacityFactor( config['opacity_factor'] );
            }

            if(config['color_factor'] != undefined) {
                me._core.setColorFactor( config['color_factor'] );
            }

            if(config['tf_colors'] != undefined) {
                me._core.setTransferFunctionByColors( config['tf_colors'] );
            }

            if(config['background'] != undefined) {
                me._core.setBackgroundColor( config['background'] );
            }

            if(config['auto_steps'] != undefined) {
                me.setAutoStepsOn( config['auto_steps'] );
            }

            if(config['axis'] != undefined) {
                me.setAxis( config['axis'] );
            }

            if(config['absorption_mode'] != undefined) {
                me._core.setAbsorptionMode( config['absorption_mode'] );
            }

            if(config['indexOfImage'] != undefined) {
                me._core.setIndexOfImage( config['indexOfImage'] );
                
            }

            //if(config['render_size'] != undefined) {
            //    me._render.setSize( config['render_size'][0], config['render_size'][1] );
            //}

            if(config['render_canvas_size'] != undefined) {
                me.setRenderCanvasSize( config['render_canvas_size'][0], config['render_canvas_size'][1] );
            }
            me._needRedraw = true;
        };

        me.uploadConfig = function(path, onLoad, onError) {
            var xmlhttp;

            if (window.XMLHttpRequest) {
                // code for IE7+, Firefox, Chrome, Opera, Safari
                xmlhttp = new XMLHttpRequest();
            } else {
                // code for IE6, IE5
                xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
            }

            xmlhttp.onreadystatechange = function() {
                if (xmlhttp.readyState == XMLHttpRequest.DONE ) {
                    if(xmlhttp.status == 200){
                        var config = JSON.parse(xmlhttp.responseText);
                        me.setConfig( config );
                        if(onLoad != undefined) onLoad();
                    } else if(xmlhttp.status == 400) {
                        if(userOnError != undefined) userOnError(xmlhttp);
                    } else {
                        if(userOnError != undefined) userOnError(xmlhttp);
                    }
                }

            }

            xmlhttp.open("GET", path, true);
            xmlhttp.send();

        };

        me.getConfig = function() {
            var config = {
                "steps": me.getSteps(),
                "slices_range": me.getSlicesRange(),
                "volume_size": me.getVolumeSize(),
                "row_col": me.getRowCol(),
                "gray_min": me.getGrayMinValue(),
                "gray_max": me.getGrayMaxValue(),
                "slicemaps_paths": me.getSlicemapsPaths(),
                "opacity_factor": me.getOpacityFactor(),
                "color_factor": me.getColorFactor(),
                "absorption_mode": me.getAbsorptionMode(),
                "render_size": me.getRenderSize(),
                "render_canvas_size": me.getRenderCanvasSize(),
                "backgound": me.getClearColor(),
                "tf_path": me.getTransferFunctionAsImage().src,
                "tf_colors": me.getTransferFunctionColors(),
                "x_min": me.getGeometryDimensions()["xmin"],
                "x_max": me.getGeometryDimensions()["xmax"],
                "y_min": me.getGeometryDimensions()["ymin"],
                "y_max": me.getGeometryDimensions()["ymax"],
                "z_min": me.getGeometryDimensions()["zmin"],
                "z_max": me.getGeometryDimensions()["zmax"],
                "dom_container_id": me.getDomContainerId(),
                "auto_steps": me.isAutoStepsOn(),
                "axis": true,
            };

            return config;
        };

        me.init();

        me.setConfig(config);

        return me;

    };

    namespace.VolumeRaycaster = VolumeRaycaster;

})(window.VRC);
