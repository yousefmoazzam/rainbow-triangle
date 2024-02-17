use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::image::{Image, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::pipeline::{GraphicsPipeline, PipelineShaderStageCreateInfo, PipelineLayout};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::color_blend::{ColorBlendState, ColorBlendAttachmentState};
use vulkano::sync::{self, GpuFuture};
use vulkano::swapchain::{ColorSpace, Surface, Swapchain, SwapchainCreateInfo};
use vulkano::instance::InstanceExtensions;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    colour: [f32; 3],
}

fn main() {
    // Setup
    let event_loop = EventLoop::new();
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = setup_instance(required_extensions);
    let window = Arc::new(
        WindowBuilder::new().build(&event_loop)
            .expect("Should be able to create window object")
    );
    let surface = Surface::from_window(instance.clone(), window.clone())
        .expect("Should be able to create surface from window");
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = get_physical_device(
        instance,
        &device_extensions,
        &surface,
    );
    let (device, queues) = get_logical_device(
        physical_device.clone(),
        queue_family_index,
        device_extensions,
    );
    let queue = get_queue(queues);
    let memory_allocator = create_memory_allocator(device.clone());

    // Create swapchain
    let (mut swapchain, images) = create_swapchain(
        physical_device.clone(),
        device.clone(),
        surface.clone(),
        window.clone(),
    );

    // Create vertices of single triangle
    let vertex1 = MyVertex { position: [-0.5, -0.5], colour: [1.0, 0.0, 0.0] };
    let vertex2 = MyVertex { position: [0.5, -0.5], colour: [0.0, 1.0, 0.0] };
    let vertex3 = MyVertex { position: [0.5, 0.5], colour: [0.0, 0.0, 1.0] };
    let vertex4 = MyVertex { position: [-0.5, 0.5], colour: [1.0, 1.0, 1.0] };

    // Create buffer on host to hold vertex data purely for transferring to "device-local" GPU
    // memory (called a "staging buffer")
    let staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3, vertex4],
    ).expect("Should have been able to create staging buffer");

    // Create vertex buffer in "device-local" memory which will be the copy destination of the
    // three triangle vertices
    let vertex_buffer = Buffer::new_slice::<MyVertex>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        4,
    ).expect("Should have been able to create vertex buffer");

    // Create staging buffer to hold indices for index buffer
    let index_staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![0u16, 1, 2, 2, 3, 0],
    ).expect("Should be able to create staging buffer for indices");

    // Create device-local buffer as copy destination for the indices
    let index_buffer = Buffer::new_slice::<u16>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        6,
    ).expect("Should be able to create index buffer");

    // Create command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Copy vertex data from staging buffer to device-local buffer
    copy_from_staging_to_device(
        device.clone(),
        queue.clone(),
        &command_buffer_allocator,
        staging_buffer.clone(),
        vertex_buffer.clone(),
    );

    // Copy indices data from staging buffer to device-local buffer
    copy_from_staging_to_device(
        device.clone(),
        queue.clone(),
        &command_buffer_allocator,
        index_staging_buffer.clone(),
        index_buffer.clone(),
    );

    // Create render pass object configured to clear a single image
    let render_pass = create_render_pass(device.clone(), swapchain.clone());

    // Create a framebuffer for each image in the swapchain
    let framebuffers = create_framebuffers(images.clone(), render_pass.clone());

    // Create shader module objects
    let vertex_shader = vertex_shaders::load(device.clone())
        .expect("Should be able to create shader module for vertex shader");
    let fragment_shader = fragment_shaders::load(device.clone())
        .expect("Should be able to create shader module for fragment shader");

    // Create viewport
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };

    // Create graphics pipeline object
    let graphics_pipeline = create_graphics_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        render_pass.clone(),
        viewport.into(),
    );

    // Record commands to builder
    let colour: [f32; 4] = [0.80, 0.97, 1.00, 1.00];

    // Create command buffer, one associated with each framebuffer
    let command_buffers = create_command_buffers(
        &command_buffer_allocator,
        queue.queue_family_index(),
        colour,
        framebuffers.clone(),
        vertex_buffer.clone(),
        index_buffer.clone(),
        graphics_pipeline.clone(),
    );

    // Create event loop which produces the window that Vulkan can use to display frames
    event_loop.run(|event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            _ => ()
        }
    });
}

fn setup_instance(extensions: InstanceExtensions) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No Vulkan library installed");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: extensions,
            ..Default::default()
        },
    ).expect("Failed to create Vulkan instance");
    instance
}

fn get_physical_device(
    instance: Arc<Instance>,
    extensions: &DeviceExtensions,
    surface: &Surface,
) -> (Arc<PhysicalDevice>, u32) {
    instance.enumerate_physical_devices()
        .expect("Couldn't find any physical devices")
        .filter(|d| d.supported_extensions().contains(extensions))
        .filter_map(|d| {
            d.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && d.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (d, q as u32))
        })
        .min_by_key(|(d, _)| match d.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("Unable to find suitable device")
}

fn get_logical_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    extensions: DeviceExtensions,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![
                QueueCreateInfo { queue_family_index, ..Default::default() },
            ],
            enabled_extensions: extensions,
            ..Default::default()
        },
    ).expect("Unable to create logical device from physical device");
    (device, queues)
}

fn get_queue(mut queues: impl ExactSizeIterator<Item = Arc<Queue>>) -> Arc<Queue> {
    let queue = queues.next().expect("Should have had 1 queue in the iterator of queues");
    queue
}

fn create_memory_allocator(device: Arc<Device>) -> Arc<StandardMemoryAllocator> {
    let allocator = Arc::new(
        StandardMemoryAllocator::new_default(device)
    );
    allocator
}

fn create_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    let render_pass = vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    ).expect("Should have been able to create render pass");
    render_pass
}

fn create_framebuffers(
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images.iter().map(|image| {
        Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    ImageView::new_default(image.clone())
                        .expect("Should be able to wrap swapchain image in an image view")
                ],
                ..Default::default()
            },
        ).expect("Should be able to create framebuffer to contain single attachment")
    }).collect()
}

fn create_command_buffers(
    allocator: &StandardCommandBufferAllocator,
    queue_family_index: u32,
    colour: [f32; 4],
    framebuffers: Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    index_buffer: Subbuffer<[u16]>,
    graphics_pipeline: Arc<GraphicsPipeline>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                allocator,
                queue_family_index,
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("Should have been able to create command buffer builder");

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some(colour.into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .expect("Should be able to configure image clearing in render pass")
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .expect("Should be able to bind graphics pipeline object")
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .expect("Should be able to bind single vertex buffer")
                .bind_index_buffer(index_buffer.clone())
                .expect("Should be able to bind index buffer")
                .draw_indexed(6, 1, 0, 0, 0)
                .expect("Should be able to draw vertices")
                .end_render_pass(SubpassEndInfo::default())
                .expect("Should be able to configure end of render pass");

            builder.build().expect("Should be able to build command buffer from builder")
        })
        .collect()
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vertex_shader.entry_point("main")
        .expect("Expected main() function in vertex shader");
    let fs = fragment_shader.entry_point("main")
        .expect("Expected main() function in fragment shader");

    // TODO: Learn more, not much is said in vulkano triangle tutorial about this
    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .expect("Should be able to get vertex input state from vertex struct");

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // TODO: Learn more, not much is said in vulkano triangle tutorial about this
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone())
        .expect("Should be able to create pipeline layout info")
    ).expect("Should be able to create pipeline layout");

    let subpass = Subpass::from(render_pass, 0)
        .expect("Should be able to create single subpass");

    let graphics_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            // Need the `into_iter().collect()` to get a `smallvec::SmallVec`, not sure why can't
            // create directly though
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                // Need the `into_iter().collect()` to get a `smallvec::SmallVec`, not sure why
                // can't create directly though
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        },
    ).expect("Should be able to create graphics pipeline object");

    graphics_pipeline
}

fn copy_from_staging_to_device<T>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    allocator: &StandardCommandBufferAllocator,
    staging_buffer: Subbuffer<[T]>,
    device_buffer: Subbuffer<[T]>,
) {
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("Should be able to create command buffer for copy command");

    builder
        .copy_buffer(CopyBufferInfo::buffers(staging_buffer, device_buffer))
        .expect("Should be able to record copy command");

    let command_buffer = builder.build()
        .expect("Should be able to build command buffer for copying");

    let future = sync::now(device.clone())
        .then_execute(queue, command_buffer)
        .expect("Should be able to submit command buffer w/ copy command")
        .then_signal_fence_and_flush()
        .expect("Should be able to ask for fence for when copy command has completed");

    future.wait(None).expect("Should be able to wait for fence from future");
}

fn create_swapchain(
    physical_device: Arc<PhysicalDevice>,
    logical_device: Arc<Device>,
    surface: Arc<Surface>,
    window: Arc<Window>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let dimensions = window.inner_size();

    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("Should be able to get capabilities of surface");
    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .expect("Should be able to get a 'composite alpha mode'");

    let (image_format, image_colour_space) = physical_device
            .surface_formats(&surface, Default::default())
            .expect("Should be able to get formats of surface the physical device supports")
            .into_iter()
            .min_by_key(|(_, colour_space)| match colour_space {
                ColorSpace::SrgbNonLinear => 0,
                _ => 1,
            })
            .expect("Should be able to pick a format + colour space for swapchain images");

    let mut image_count = caps.min_image_count + 1;
    if let Some(val) = caps.max_image_count {
        image_count = val;
    }

    Swapchain::new(
        logical_device,
        surface,
        SwapchainCreateInfo {
            min_image_count: image_count,
            image_format: image_format,
            image_extent: dimensions.into(),
            image_color_space: image_colour_space,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: composite_alpha,
            ..Default::default()
        },
    ).expect("Should be able to create swapchain")
}

mod vertex_shaders {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec3 colour;

            layout(location = 0) out vec3 fragColour;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                fragColour = colour;
            }
        ",
    }
}

mod fragment_shaders {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec3 fragColour;
            layout(location = 0) out vec4 outColour;

            void main() {
                outColour = vec4(fragColour, 1.0);
            }
        ",
    }
}
