use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::format::Format;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
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

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn main() {
    // Setup
    let instance = setup_instance();
    let physical_device = get_physical_device(instance);
    let (device, queues) = get_logical_device(physical_device);
    let queue = get_queue(queues);
    let memory_allocator = create_memory_allocator(device.clone());

    // Image creation
    let height = 1024;
    let width = 1024;
    let image = create_image(memory_allocator.clone(), height, width);

    // Create vertices of single triangle
    let vertex1 = MyVertex { position: [0.0, -0.5] };
    let vertex2 = MyVertex { position: [0.5, 0.5] };
    let vertex3 = MyVertex { position: [-0.5, 0.5] };

    // Create vertex buffer and put the triangle vertices in it
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    ).expect("Should have been able to create vertex buffer");

    // Create render pass object configured to clear a single image
    let render_pass = create_render_pass(device.clone());

    // Create framebuffer that contains the single image
    let framebuffer = wrap_image_in_framebuffer(image.clone(), render_pass.clone());

    // Create command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Create command buffer builder
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("Should have been able to create command buffer builder");

    // Record commands to builder
    let colour: [f32; 4] = [0.80, 0.97, 1.00, 1.00];
    configure_command_buffer_builder(&mut builder, colour, framebuffer);

    // Create shader module objects
    let vertex_shader = vertex_shaders::load(device.clone())
        .expect("Should be able to create shader module for vertex shader");
    let fragment_shader = fragment_shaders::load(device.clone())
        .expect("Should be able to create shader module for fragment shader");

    // Create viewport
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [height as f32, width as f32],
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
}

fn setup_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No Vulkan library installed");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("Failed to create Vulkan instance");
    instance
}

fn get_physical_device(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
    let mut physical_devices = instance.enumerate_physical_devices()
        .expect("Couldn't find any physical devices");
    let physical_device = physical_devices.next().expect("No device found");
    physical_device
}

fn get_logical_device(
    physical_device: Arc<PhysicalDevice>,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![
                QueueCreateInfo { queue_family_index, ..Default::default() },
            ],
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

fn create_image(
    allocator: Arc<StandardMemoryAllocator>,
    height: u32,
    width: u32,
) -> Arc<Image> {
    let image = Image::new(
        allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [height, width, 1],
            usage: ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).expect("Should've been able to create a single 2D image");
    image
}

fn create_render_pass(device: Arc<Device>) -> Arc<RenderPass> {
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
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

fn wrap_image_in_framebuffer(
    image: Arc<Image>,
    render_pass: Arc<RenderPass>,
) -> Arc<Framebuffer> {
    let view = ImageView::new_default(image)
        .expect("Should have been able to create an image view");
    let framebuffer = Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    ).expect("Should have been able to create framebuffer");
    framebuffer
}

fn configure_command_buffer_builder(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    colour: [f32; 4],
    framebuffer: Arc<Framebuffer>,
) {
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(colour.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("Should be able to configure image clearing in render pass")
        .end_render_pass(SubpassEndInfo::default())
        .expect("Should be able to configure end of render pass");
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

mod vertex_shaders {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fragment_shaders {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(0.0, 0.0, 1.0, 1.0);
            }
        ",
    }
}
