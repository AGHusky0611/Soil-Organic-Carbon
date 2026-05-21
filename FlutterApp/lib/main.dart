import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

enum AnalyzeMode { combined, classification, soc }

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SoilApp());
}

class SoilApp extends StatelessWidget {
  const SoilApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(seedColor: Colors.green);
    return MaterialApp(
      title: 'Soil Organic Carbon',
      theme: ThemeData(colorScheme: colorScheme, useMaterial3: true),
      home: const SoilHomePage(),
    );
  }
}

class SoilHomePage extends StatefulWidget {
  const SoilHomePage({super.key});

  @override
  State<SoilHomePage> createState() => _SoilHomePageState();
}

class _SoilHomePageState extends State<SoilHomePage> {
  final _picker = ImagePicker();
  final _apiController = TextEditingController(text: 'http://10.0.2.2:8000');

  AnalyzeMode _mode = AnalyzeMode.combined;
  File? _imageFile;
  Map<String, dynamic>? _result;
  bool _isLoading = false;
  String? _error;

  @override
  void dispose() {
    _apiController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) {
      return;
    }

    setState(() {
      _imageFile = File(image.path);
      _result = null;
      _error = null;
    });
  }

  Future<void> _openCamera() async {
    final captured = await Navigator.of(
      context,
    ).push<File?>(MaterialPageRoute(builder: (_) => const CameraCapturePage()));
    if (captured == null) {
      return;
    }

    setState(() {
      _imageFile = captured;
      _result = null;
      _error = null;
    });
  }

  String _endpointForMode(AnalyzeMode mode) {
    switch (mode) {
      case AnalyzeMode.classification:
        return '/classify';
      case AnalyzeMode.soc:
        return '/soc';
      case AnalyzeMode.combined:
        return '/analyze';
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      setState(() {
        _error = 'Pick an image first.';
      });
      return;
    }

    final baseUrl = _apiController.text.trim();
    if (baseUrl.isEmpty) {
      setState(() {
        _error = 'Enter the API base URL.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
    });

    try {
      final uri = Uri.parse('$baseUrl${_endpointForMode(_mode)}');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _result = data;
        });
      } else {
        setState(() {
          _error = 'Server error: ${response.statusCode} ${response.body}';
        });
      }
    } catch (exc) {
      setState(() {
        _error = 'Request failed: $exc';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Soil Organic Carbon')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _apiController,
                keyboardType: TextInputType.url,
                decoration: const InputDecoration(
                  labelText: 'API Base URL',
                  helperText: 'Android emulator uses http://10.0.2.2:8000',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 16),
              SegmentedButton<AnalyzeMode>(
                segments: const [
                  ButtonSegment(
                    value: AnalyzeMode.combined,
                    label: Text('Combined'),
                  ),
                  ButtonSegment(
                    value: AnalyzeMode.classification,
                    label: Text('Classification'),
                  ),
                  ButtonSegment(
                    value: AnalyzeMode.soc,
                    label: Text('SOC Only'),
                  ),
                ],
                selected: {_mode},
                onSelectionChanged: (selection) {
                  setState(() {
                    _mode = selection.first;
                    _result = null;
                    _error = null;
                  });
                },
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _pickImage,
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text('Gallery'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _openCamera,
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text('Camera'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              if (_imageFile != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(
                    _imageFile!,
                    height: 240,
                    fit: BoxFit.cover,
                  ),
                ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _isLoading ? null : _analyze,
                icon: const Icon(Icons.analytics_outlined),
                label: Text(_isLoading ? 'Analyzing...' : 'Analyze'),
              ),
              if (_error != null) ...[
                const SizedBox(height: 16),
                Text(
                  _error!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ],
              if (_result != null) ...[
                const SizedBox(height: 20),
                _ResultCard(result: _result!, mode: _mode),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result, required this.mode});

  final Map<String, dynamic> result;
  final AnalyzeMode mode;

  @override
  Widget build(BuildContext context) {
    final soc = result['soc'] as Map<String, dynamic>?;
    final isSoil = result['is_soil'] == true;
    final showSoc = mode != AnalyzeMode.classification;

    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Class: ${result['predicted_class']}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              'Confidence: ${(result['confidence'] as num).toStringAsFixed(3)}',
            ),
            Text('Margin: ${(result['margin'] as num).toStringAsFixed(3)}'),
            Text('Is Soil: $isSoil'),
            if (showSoc) ...[
              const SizedBox(height: 12),
              if (soc != null) ...[
                Text(
                  'SOC: ${(soc['percent'] as num).toStringAsFixed(2)}%',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text('g/kg: ${(soc['g_per_kg'] as num).toStringAsFixed(2)}'),
                Text('Category: ${soc['category']}'),
                const SizedBox(height: 8),
                Text('${soc['note']}'),
              ] else
                Text(
                  isSoil
                      ? 'SOC: model not available'
                      : 'SOC blocked: image not classified as soil',
                ),
            ],
          ],
        ),
      ),
    );
  }
}

class CameraCapturePage extends StatefulWidget {
  const CameraCapturePage({super.key});

  @override
  State<CameraCapturePage> createState() => _CameraCapturePageState();
}

class _CameraCapturePageState extends State<CameraCapturePage> {
  CameraController? _controller;
  Future<void>? _initFuture;
  String? _error;
  bool _isCapturing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _error = 'No cameras available.';
        });
        return;
      }

      final selected = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selected,
        ResolutionPreset.high,
        enableAudio: false,
      );
      setState(() {
        _controller = controller;
        _initFuture = controller.initialize();
      });
    } catch (exc) {
      setState(() {
        _error = 'Camera error: $exc';
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null || _isCapturing) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      await _initFuture;
      final file = await controller.takePicture();
      if (!mounted) {
        return;
      }
      Navigator.of(context).pop(File(file.path));
    } catch (exc) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Capture failed: $exc';
        _isCapturing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(title: const Text('Camera')),
      body: _error != null
          ? Center(child: Text(_error!))
          : controller == null
          ? const Center(child: CircularProgressIndicator())
          : FutureBuilder<void>(
              future: _initFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const Center(child: CircularProgressIndicator());
                }
                return Column(
                  children: [
                    Expanded(child: CameraPreview(controller)),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: _isCapturing ? null : _capture,
                          icon: const Icon(Icons.camera),
                          label: Text(
                            _isCapturing ? 'Capturing...' : 'Capture photo',
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
    );
  }
}
