def normalize_1d_array(arr):
    "Normalize 1d array"
    assert arr.ndim == 1
    result = None
    if np.linalg.norm(arr) == 0:
        result = arr
    else:
        result = arr / np.linalg.norm(arr)
    return result

def normalize_3col_array(arr):
    "Normalize 3 column array"
    assert arr.shape[1] == 3
    assert arr.ndim == 2
    normal = np.copy(arr)
    normal[:, 0] = normalize_1d_array(normal[:, 0])
    normal[:, 1] = normalize_1d_array(normal[:, 1])
    normal[:, 2] = normalize_1d_array(normal[:, 2])
    return normal


def get_vector_dot(arr1, arr2):
    "Get vector dot product for 2 matrices"
    assert arr1.shape == arr2.shape
    newarr = np.sum(arr1 * arr2, axis=1, dtype=np.float32)
    return newarr


class ImageArray:
    "Image array have some additional properties besides np.ndarray"

    def __init__(self, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        self.image = image

    @property
    def norm_coordinates(self):
        "Get normalized coordinates of the image pixels"
        # pdb.set_trace()
        rownb, colnb = self.image.shape[0], self.image.shape[1]
        norm = np.empty_like(self.coordinates, dtype=np.float32)
        norm[:, 0] = self.coordinates[:, 0] / rownb
        norm[:, 1] = self.coordinates[:, 1] / colnb
        return norm

    @property
    def norm_image(self):
        "Get normalized image with pixel values divided by 255"
        return self.image / 255.0

    @property
    def coordinates(self):
        "Coordinates of the image pixels"
        rownb, colnb = self.image.shape[:2]
        coords = [[(row, col) for col in range(colnb)] for row in range(rownb)]
        coordarray = np.array(coords)
        return coordarray.reshape((-1, 2))

    @property
    def arrshape(self):
        "get array shape"
        return self.image.shape

    @property
    def flatarr(self):
        "get flattened array"
        return self.image.flatten()


def interpolateImage(imarr: ImageArray):
    "Interpolate image array"
    imshape = imarr.image.shape
    newimage = imarr.image.flatten()
    newimage = np.uint8(np.interp(newimage,
                                  (newimage.min(),
                                   newimage.max()),
                                  (0, 255))
                        )
    newimage = newimage.reshape(imshape)
    return ImageArray(newimage)


class LightSource:
    "Simple implementation of a light source"

    def __init__(self,
                 x=1.0,  # x
                 y=1.0,  # y
                 z=20.0,  # light source distance: 0 to make it at infinity
                 intensity=1.0,  # I_p
                 ambient_intensity=1.0,  # I_a
                 ambient_coefficient=0.000000002,  # k_a
                 ):
        "light source"
        self.x = x
        self.y = y
        if z is not None:
            assert isinstance(z, float)
        self.z = z
        self.intensity = intensity
        self.ambient_intensity = ambient_intensity  # I_a
        self.ambient_coefficient = ambient_coefficient  # k_a
        # k_a can be tuned if the material is known

    def copy(self):
        "copy self"
        return LightSource(x=self.x,
                           y=self.y,
                           z=self.z,
                           intensity=self.intensity,
                           light_power=self.power)


class ChannelShader:
    "Shades channels"

    def __init__(self,
                 coordarr: np.ndarray,
                 light_source: LightSource,  # has I_a, I_p, k_a
                 surface_normal: np.ndarray,
                 color: np.ndarray,  # they are assumed to be O_d and O_s
                 spec_coeff=0.1,  # k_s
                 spec_color=1.0,  # O_s: obj specular color. It can be
                 # optimized with respect to surface material
                 screen_gamma=2.2,
                 diffuse_coeff=0.008,  # k_d
                 # a good value is between 0.007 and 0.1
                 attenuation_c1=1.0,  # f_attr c1
                 attenuation_c2=0.0,  # f_attr c2 d_L coefficient
                 attenuation_c3=0.0,  # f_attr c3 d_L^2 coefficient
                 shininess=20.0  # n
                 ):
        self.light_source = light_source
        self.light_intensity = self.light_source.intensity  # I_p
        self.ambient_coefficient = self.light_source.ambient_coefficient  # k_a
        self.ambient_intensity = self.light_source.ambient_intensity  # I_a
        self.coordarr = coordarr
        self.surface_normal = np.copy(surface_normal)
        self.screen_gamma = screen_gamma
        self.shininess = shininess
        self.diffuse_coeff = diffuse_coeff  # k_d
        # self.diffuse_color = normalize_1d_array(color)  # O_d: obj diffuse color
        self.diffuse_color = color  # O_d: obj diffuse color
        self.spec_color = spec_color  # O_s
        self.spec_coeff = spec_coeff  # k_s: specular coefficient
        self.att_c1 = attenuation_c1
        self.att_c2 = attenuation_c2
        self.att_c3 = attenuation_c3

    def copy(self):
        return ChannelShader(coordarr=np.copy(self.coordarr),
                             light_source=self.light_source.copy(),
                             surface_normal=np.copy(self.surface_normal),
                             color=np.copy(self.diffuse_color))

    @property
    def distance(self):
        yarr = self.coordarr[:, 0]  # row nb
        xarr = self.coordarr[:, 1]  # col nb
        xdist = (self.light_source.x - xarr)**2
        ydist = (self.light_source.y - yarr)**2
        return xdist + ydist

    @property
    def light_direction(self):
        "get light direction matrix (-1, 3)"
        yarr = self.coordarr[:, 0]
        xarr = self.coordarr[:, 1]
        xdiff = self.light_source.x - xarr
        ydiff = self.light_source.y - yarr
        light_matrix = np.zeros((self.coordarr.shape[0], 3))
        light_matrix[:, 0] = ydiff
        light_matrix[:, 1] = xdiff
        light_matrix[:, 2] = self.light_source.z
        # light_matrix[:, 2] = 0.0
        # pdb.set_trace()
        return light_matrix

    @property
    def light_attenuation(self):
        """
        Implementing from Foley JD 1996, p. 726

        f_att : light source attenuation function:
        f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
        """
        second = self.att_c2 * self.distance
        third = self.att_c3 * self.distance * self.distance
        result = self.att_c1 + second + third
        result = 1 / result
        return np.where(result < 1, result, 1)

    @property
    def normalized_light_direction(self):
        "Light Direction matrix normalized"
        return normalize_3col_array(self.light_direction)

    @property
    def normalized_surface_normal(self):
        return normalize_3col_array(self.surface_normal)

    @property
    def costheta(self):
        "set costheta"
        # pdb.set_trace()
        costheta = get_vector_dot(
            arr1=self.normalized_light_direction,
            arr2=self.normalized_surface_normal)
        # products of vectors
        # costheta = np.abs(costheta)  # as per (Foley J.D, et.al. 1996, p. 724)
        costheta = np.where(costheta > 0, costheta, 0)
        return costheta

    @property
    def ambient_term(self):
        "Get the ambient term I_a * k_a * O_d"
        term = self.ambient_coefficient * self.ambient_intensity
        term *= self.diffuse_color
        # pdb.set_trace()
        return term

    @property
    def view_direction(self):
        "Get view direction"
        # pdb.set_trace()
        cshape = self.coordarr.shape
        coord = np.zeros((cshape[0], 3))  # x, y, z
        coord[:, :2] = -self.coordarr
        coord[:, 2] = 0.0  # viewer at infinity
        coord = normalize_3col_array(coord)
        return coord

    @property
    def half_direction(self):
        "get half direction"
        # pdb.set_trace()
        arr = self.view_direction + self.normalized_light_direction
        return normalize_3col_array(arr)

    @property
    def spec_angle(self):
        "get spec angle"
        specAngle = get_vector_dot(
            arr1=self.half_direction,
            arr2=self.normalized_surface_normal)
        return np.where(specAngle > 0.0, specAngle, 0.0)

    @property
    def specular(self):
        return self.spec_angle ** self.shininess

    @property
    def channel_color_blinn_phong(self):
        """compute new channel color intensities
        Implements: Foley J.D. 1996 p. 730 - 731, variation on equation 16.15
        """
        second = 1.0  # added for structuring code in this fashion, makes
        # debugging easier
        # lambertian terms
        second *= self.diffuse_coeff  # k_d
        second *= self.costheta  # (N \cdot L)
        second *= self.light_intensity  # I_p
        # adding phong terms
        second *= self.light_attenuation  # f_attr
        second *= self.diffuse_color  # O_d
        third = 1.0
        third *= self.spec_color  # O_s
        third *= self.specular  # (N \cdot H)^n
        third *= self.spec_coeff  # k_s
        result = 0.0
        #
        result += self.ambient_term  # I_a × k_a × O_d
        result += second
        result += third
        # pdb.set_trace()
        return result
